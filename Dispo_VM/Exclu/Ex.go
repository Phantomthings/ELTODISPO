// ELTO borne_id=0 dispo_blocs_ac_sites
package main

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"os"
	"regexp"
	"sync"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

const NumWorkers = 10

var (
	responsabilityStatus = map[string]int{
		"Elto":   2,
		"Enedis": 3,
	}
)

type DbConfig struct {
	User     string
	Password string
	Host     string
	Port     string
	Database string
}

type Row struct {
	ID            int
	Site          string
	PdcID         string
	DateDebut     time.Time
	DateFin       sql.NullTime
	EstDisponible int
	Cause         sql.NullString
	ICPC          sql.NullString
}

type Ticket struct {
	IDIndispo      int
	ProjetGlobal   sql.NullString
	ProjectID      sql.NullString
	BorneID        sql.NullString
	StartTime      time.Time
	EndTime        sql.NullTime
	Responsability sql.NullString
	TicketID       sql.NullInt64
	IDJournal      sql.NullInt64
	IDTask         sql.NullInt64
}

type TimeSegment struct {
	Start  time.Time
	End    time.Time
	Ticket *Ticket
}

type Stats struct {
	Updated       int
	Tracked       int
	WithoutTicket int
	SkippedBorne  int
	Split         int
	Deleted       int
	SitesUpdated  int
}

type TableJob struct {
	Table string
}

type Logger struct {
	verbose bool
	mu      sync.Mutex
}

func (l *Logger) Info(format string, v ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	log.Printf("[INFO] "+format, v...)
}

func (l *Logger) Debug(format string, v ...interface{}) {
	if l.verbose {
		l.mu.Lock()
		defer l.mu.Unlock()
		log.Printf("[DEBUG] "+format, v...)
	}
}

func (l *Logger) Warning(format string, v ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	log.Printf("[WARNING] "+format, v...)
}

func (l *Logger) Error(format string, v ...interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	log.Printf("[ERROR] "+format, v...)
}

var logger *Logger

func getDbConfig() DbConfig {
	return DbConfig{
		User:     getEnv("MYSQL_USER", "AdminNidec"),
		Password: getEnv("MYSQL_PASSWORD", "u6Ehe987XBSXxa4"),
		Host:     getEnv("MYSQL_HOST", "141.94.31.144"),
		Port:     getEnv("MYSQL_PORT", "3306"),
		Database: getEnv("MYSQL_DB", "indicator"),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func createDbConnection(config DbConfig) (*sql.DB, error) {
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?charset=utf8mb4&parseTime=true",
		config.User, config.Password, config.Host, config.Port, config.Database)

	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return nil, err
	}

	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(10)
	db.SetConnMaxLifetime(time.Hour)

	if err := db.Ping(); err != nil {
		return nil, err
	}

	return db, nil
}

func listPdcTables(db *sql.DB, database string) ([]string, error) {
	query := `
		SELECT TABLE_NAME
		FROM information_schema.tables
		WHERE TABLE_SCHEMA = ?
		  AND TABLE_NAME REGEXP '^dispo_pdc_[nN]?[0-9]+_[0-9]+(_[0-9]+)?$'
		ORDER BY TABLE_NAME
	`

	rows, err := db.Query(query, database)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tables []string
	for rows.Next() {
		var table string
		if err := rows.Scan(&table); err != nil {
			return nil, err
		}
		tables = append(tables, table)
	}

	return tables, rows.Err()
}

func extractPdcFromTableName(tableName string) string {
	re := regexp.MustCompile(`^dispo_pdc_[nN]?(\d+)_`)
	matches := re.FindStringSubmatch(tableName)
	if len(matches) > 1 {
		return matches[1]
	}
	return ""
}

func parseDate(dateStr string) (time.Time, error) {
	if dateStr == "" {
		return time.Time{}, fmt.Errorf("date vide")
	}
	return time.Parse("2006-01-02 15:04:05", dateStr)
}

func loadRowsToProcess(db *sql.DB, table, mode string, start, end *time.Time) ([]Row, error) {
	var query string
	var args []interface{}

	switch mode {
	case "period":
		if start == nil || end == nil {
			return nil, fmt.Errorf("le mode période requiert --start-date et --end-date")
		}
		query = fmt.Sprintf(`
			SELECT id, site, pdc_id, date_debut, date_fin, est_disponible, cause, ICPC
			FROM %s
			WHERE est_disponible = 0
			  AND date_debut <= ?
			  AND (date_fin IS NULL OR date_fin >= ?)
			ORDER BY date_debut
		`, quoteIdentifier(table))
		args = []interface{}{end, start}

	case "full":
		query = fmt.Sprintf(`
			SELECT id, site, pdc_id, date_debut, date_fin, est_disponible, cause, ICPC
			FROM %s
			WHERE est_disponible = 0
			ORDER BY date_debut
		`, quoteIdentifier(table))

	default: // incremental
		query = fmt.Sprintf(`
			SELECT id, site, pdc_id, date_debut, date_fin, est_disponible, cause, ICPC
			FROM %s t
			WHERE (
				  (t.est_disponible = 0 AND NOT EXISTS (
					  SELECT 1
					  FROM exclusion_tracking et
					  WHERE et.table_source = ?
						AND et.indispo_id = t.id
						AND et.date_debut = t.date_debut
				  ))
				  OR
				  t.id IN (
					  SELECT et.indispo_id
					  FROM exclusion_tracking et
					  WHERE et.table_source = ?
						AND et.is_ongoing = TRUE
				  )
			  )
			ORDER BY date_debut
		`, quoteIdentifier(table))
		args = []interface{}{table, table}
	}

	rows, err := db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []Row
	for rows.Next() {
		var r Row
		if err := rows.Scan(&r.ID, &r.Site, &r.PdcID, &r.DateDebut, &r.DateFin, &r.EstDisponible, &r.Cause, &r.ICPC); err != nil {
			return nil, err
		}
		result = append(result, r)
	}

	return result, rows.Err()
}

func findAllOverlappingTickets(db *sql.DB, site, pdcID string, start, end time.Time) ([]Ticket, error) {
	query := `
		SELECT id_indispo, Projet_global, project_id, borne_id, starttime, endtime,
			   responsability, ticket_id, id_journal, id_task
		FROM Exclu_ticket
		WHERE (project_id = ? OR Projet_global = ?)
		  AND (borne_id = ? OR borne_id = 'all' OR borne_id = '0')
		  AND starttime < ?
		  AND (endtime IS NULL OR endtime > ?)
		ORDER BY starttime, (borne_id = ?) DESC
	`

	rows, err := db.Query(query, site, site, pdcID, end, start, pdcID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tickets []Ticket
	for rows.Next() {
		var ticket Ticket
		if err := rows.Scan(
			&ticket.IDIndispo,
			&ticket.ProjetGlobal,
			&ticket.ProjectID,
			&ticket.BorneID,
			&ticket.StartTime,
			&ticket.EndTime,
			&ticket.Responsability,
			&ticket.TicketID,
			&ticket.IDJournal,
			&ticket.IDTask,
		); err != nil {
			return nil, err
		}
		tickets = append(tickets, ticket)
	}

	return tickets, rows.Err()
}

func splitIntoSegments(start, end time.Time, tickets []Ticket) []TimeSegment {
	if len(tickets) == 0 {
		return []TimeSegment{{Start: start, End: end, Ticket: nil}}
	}

	type timePoint struct {
		t       time.Time
		isStart bool
		ticket  *Ticket
	}

	var points []timePoint
	points = append(points, timePoint{t: start, isStart: true, ticket: nil})
	points = append(points, timePoint{t: end, isStart: false, ticket: nil})

	for i := range tickets {
		ticket := &tickets[i]
		ticketStart := ticket.StartTime
		ticketEnd := end
		if ticket.EndTime.Valid {
			ticketEnd = ticket.EndTime.Time
		}

		if ticketStart.Before(start) {
			ticketStart = start
		}
		if ticketEnd.After(end) {
			ticketEnd = end
		}

		if ticketStart.Before(end) && ticketEnd.After(start) {
			points = append(points, timePoint{t: ticketStart, isStart: true, ticket: ticket})
			points = append(points, timePoint{t: ticketEnd, isStart: false, ticket: ticket})
		}
	}

	for i := 0; i < len(points); i++ {
		for j := i + 1; j < len(points); j++ {
			if points[j].t.Before(points[i].t) {
				points[i], points[j] = points[j], points[i]
			}
		}
	}

	var segments []TimeSegment
	activeTickets := make(map[*Ticket]bool)
	lastTime := start

	for _, point := range points {
		if point.t.After(lastTime) {
			var bestTicket *Ticket
			for ticket := range activeTickets {
				if bestTicket == nil {
					bestTicket = ticket
				} else {
					if ticket.BorneID.Valid && ticket.BorneID.String != "all" && ticket.BorneID.String != "0" {
						if !bestTicket.BorneID.Valid || bestTicket.BorneID.String == "all" || bestTicket.BorneID.String == "0" {
							bestTicket = ticket
						}
					}
				}
			}

			segments = append(segments, TimeSegment{
				Start:  lastTime,
				End:    point.t,
				Ticket: bestTicket,
			})
		}

		if point.ticket != nil {
			if point.isStart {
				activeTickets[point.ticket] = true
			} else {
				delete(activeTickets, point.ticket)
			}
		}

		lastTime = point.t
	}

	return segments
}

func computeStatus(responsability sql.NullString) *int {
	if !responsability.Valid || responsability.String == "" {
		val := 1
		return &val
	}

	resp := responsability.String
	if resp == "Nidec" {
		return nil
	}

	if status, ok := responsabilityStatus[resp]; ok {
		return &status
	}

	val := 1
	return &val
}

func chooseTicketIdentifier(ticket *Ticket) *int64 {
	if ticket.TicketID.Valid {
		return &ticket.TicketID.Int64
	}
	if ticket.IDJournal.Valid {
		return &ticket.IDJournal.Int64
	}
	if ticket.IDTask.Valid {
		return &ticket.IDTask.Int64
	}
	return nil
}

func hashSig(site, equip string, start, end time.Time, state int, cause *string) string {
	causeStr := "OK"
	if cause != nil {
		causeStr = *cause
	}
	src := fmt.Sprintf("%s|%s|%s|%s|%d|%s",
		site, equip, start.UTC().Format(time.RFC3339), end.UTC().Format(time.RFC3339), state, causeStr)
	sum := sha256.Sum256([]byte(src))
	return hex.EncodeToString(sum[:])
}

func upsertTracking(db *sql.DB, table string, row Row, ticketID *int64, applied bool, endEffective time.Time, isOngoing bool) error {
	query := `
		INSERT INTO exclusion_tracking (
			table_source, indispo_id, site, pdc_id, date_debut, date_fin,
			last_processed_at, ticket_id, exclusion_applied, is_ongoing
		)
		VALUES (?, ?, ?, ?, ?, ?, NOW(), ?, ?, ?)
		ON DUPLICATE KEY UPDATE
			site = VALUES(site),
			pdc_id = VALUES(pdc_id),
			date_fin = VALUES(date_fin),
			last_processed_at = VALUES(last_processed_at),
			ticket_id = VALUES(ticket_id),
			exclusion_applied = VALUES(exclusion_applied),
			is_ongoing = VALUES(is_ongoing)
	`

	var dateFin interface{}
	if isOngoing {
		dateFin = nil
	} else {
		dateFin = endEffective
	}

	_, err := db.Exec(query,
		table,
		row.ID,
		row.Site,
		row.PdcID,
		row.DateDebut,
		dateFin,
		ticketID,
		applied,
		isOngoing,
	)

	return err
}

func insertSplitBlock(db *sql.DB, tx *sql.Tx, table string, originalRow Row, segment TimeSegment, status int, ticketID *int64, batchID string) error {
	var cause sql.NullString
	if status == 0 && originalRow.Cause.Valid {
		cause = originalRow.Cause
	}

	causePtr := (*string)(nil)
	if cause.Valid {
		causePtr = &cause.String
	}

	hash := hashSig(originalRow.Site, originalRow.PdcID, segment.Start, segment.End, status, causePtr)

	icpc := ""
	if originalRow.ICPC.Valid {
		icpc = originalRow.ICPC.String
	}

	query := fmt.Sprintf(`
		INSERT INTO %s (
			site, pdc_id, type_label, date_debut, date_fin, 
			est_disponible, cause, ICPC, raw_point_count,
			processed_at, batch_id, hash_signature, Ticket_id
		)
		VALUES (?, ?, 'PDC', ?, ?, ?, ?, ?, ?, UTC_TIMESTAMP(), ?, ?, ?)
	`, quoteIdentifier(table))

	durationMinutes := int(segment.End.Sub(segment.Start).Minutes())

	_, err := tx.Exec(query,
		originalRow.Site,
		originalRow.PdcID,
		segment.Start,
		segment.End,
		status,
		cause,
		icpc,
		durationMinutes,
		batchID,
		hash,
		ticketID,
	)

	return err
}

// NOUVEAU: Applique l'exclusion à dispo_blocs_ac_sites pour borne_id=0
func applyExclusionToSites(db *sql.DB, site string, start, end time.Time, status int, ticketID *int64) (int, error) {
	updateQuery := `
		UPDATE dispo_blocs_ac_sites
		SET est_disponible = ?,
			Ticket_id = ?
		WHERE site = ?
		  AND date_debut < ?
		  AND date_fin > ?
		  AND est_disponible = 0
	`

	result, err := db.Exec(updateQuery, status, ticketID, site, end, start)
	if err != nil {
		return 0, err
	}

	affected, _ := result.RowsAffected()
	return int(affected), nil
}

func processTable(db *sql.DB, table, mode string, start, end *time.Time, now time.Time) (Stats, error) {
	rows, err := loadRowsToProcess(db, table, mode, start, end)
	if err != nil {
		return Stats{}, err
	}

	logger.Info("%s: %d lignes à traiter", table, len(rows))

	var stats Stats
	tablePdc := extractPdcFromTableName(table)
	if tablePdc == "" {
		logger.Warning("%s: impossible d'extraire le PDC du nom de table", table)
		return stats, nil
	}

	for _, row := range rows {
		isOngoing := !row.DateFin.Valid
		endEffective := now
		if row.DateFin.Valid {
			endEffective = row.DateFin.Time
		}

		tickets, err := findAllOverlappingTickets(db, row.Site, row.PdcID, row.DateDebut, endEffective)
		if err != nil {
			return stats, err
		}

		var validTickets []Ticket
		for _, ticket := range tickets {
			ticketBorneID := ""
			if ticket.BorneID.Valid {
				ticketBorneID = ticket.BorneID.String
			}

			// Si borne_id=0, c'est valide pour tous (sera traité séparément)
			// Si borne_id=all, c'est valide pour tous les PDC
			// Sinon, doit matcher le tablePdc
			if ticketBorneID != "" && ticketBorneID != "all" && ticketBorneID != "0" && ticketBorneID != tablePdc {
				logger.Debug("%s: id=%d ticket ignoré (borne_id=%s != table_pdc=%s)",
					table, row.ID, ticketBorneID, tablePdc)
				stats.SkippedBorne++
				continue
			}

			validTickets = append(validTickets, ticket)
		}

		segments := splitIntoSegments(row.DateDebut, endEffective, validTickets)

		if len(segments) > 1 {
			logger.Info("%s: id=%d split en %d segments (date_fin=%v, ongoing=%v)",
				table, row.ID, len(segments), row.DateFin.Valid, isOngoing)
			stats.Split++

			tx, err := db.Begin()
			if err != nil {
				return stats, err
			}

			batchID := fmt.Sprintf("SPLIT-%s-%s", time.Now().UTC().Format("20060102T150405Z"), tablePdc)
			hasExclusion := false

			for i, seg := range segments {
				segStatus := 0
				var ticketID *int64

				if seg.Ticket != nil {
					status := computeStatus(seg.Ticket.Responsability)
					if status != nil {
						segStatus = *status
						ticketID = chooseTicketIdentifier(seg.Ticket)
						hasExclusion = true

						// NOUVEAU: Si borne_id=0, appliquer à dispo_blocs_ac_sites
						if seg.Ticket.BorneID.Valid && seg.Ticket.BorneID.String == "0" {
							affected, err := applyExclusionToSites(db, row.Site, seg.Start, seg.End, segStatus, ticketID)
							if err != nil {
								logger.Warning("%s: id=%d erreur lors de l'application à dispo_blocs_ac_sites: %v", table, row.ID, err)
							} else {
								stats.SitesUpdated += affected
								logger.Debug("%s: id=%d segment %d -> %d lignes mises à jour dans dispo_blocs_ac_sites", table, row.ID, i+1, affected)
							}
						}
					}
				}

				if err := insertSplitBlock(db, tx, table, row, seg, segStatus, ticketID, batchID); err != nil {
					tx.Rollback()
					return stats, err
				}

				logger.Debug("%s: id=%d segment %d/%d [%s → %s] -> status=%d ticket=%v",
					table, row.ID, i+1, len(segments),
					seg.Start.Format("2006-01-02 15:04:05"),
					seg.End.Format("2006-01-02 15:04:05"),
					segStatus, ticketID)
			}

			deleteQuery := fmt.Sprintf(`DELETE FROM %s WHERE id = ?`, quoteIdentifier(table))
			if _, err := tx.Exec(deleteQuery, row.ID); err != nil {
				tx.Rollback()
				return stats, err
			}
			stats.Deleted++

			if err := tx.Commit(); err != nil {
				return stats, err
			}

			if hasExclusion {
				stats.Updated++
			}

			for _, seg := range segments {
				var ticketID *int64
				applied := false

				segmentIsOngoing := isOngoing

				if seg.Ticket != nil {
					status := computeStatus(seg.Ticket.Responsability)
					if status != nil {
						ticketID = chooseTicketIdentifier(seg.Ticket)
						applied = true

						if !seg.Ticket.EndTime.Valid {
							segmentIsOngoing = true
						}
					}
				}

				tempRow := Row{
					ID:        row.ID,
					Site:      row.Site,
					PdcID:     row.PdcID,
					DateDebut: seg.Start,
				}

				if !segmentIsOngoing {
					tempRow.DateFin = sql.NullTime{Time: seg.End, Valid: true}
				}

				if err := upsertTracking(db, table, tempRow, ticketID, applied, seg.End, segmentIsOngoing); err != nil {
					return stats, err
				}

				logger.Debug("%s: id=%d segment tracked [%s → %s] ongoing=%v applied=%v ticket=%v",
					table, row.ID,
					seg.Start.Format("2006-01-02 15:04:05"),
					seg.End.Format("2006-01-02 15:04:05"),
					segmentIsOngoing, applied, ticketID)
			}

		} else {
			seg := segments[0]
			applied := false
			var ticketIdentifier *int64

			segmentIsOngoing := isOngoing

			if seg.Ticket != nil {
				status := computeStatus(seg.Ticket.Responsability)
				if status != nil {
					ticketIdentifier = chooseTicketIdentifier(seg.Ticket)

					if !seg.Ticket.EndTime.Valid {
						segmentIsOngoing = true
					}

					updateQuery := fmt.Sprintf(`
						UPDATE %s
						SET est_disponible = ?,
							Ticket_id = ?
						WHERE id = ?
					`, quoteIdentifier(table))

					_, err := db.Exec(updateQuery, *status, ticketIdentifier, row.ID)
					if err != nil {
						return stats, err
					}

					// NOUVEAU: Si borne_id=0, appliquer à dispo_blocs_ac_sites
					if seg.Ticket.BorneID.Valid && seg.Ticket.BorneID.String == "0" {
						affected, err := applyExclusionToSites(db, row.Site, seg.Start, seg.End, *status, ticketIdentifier)
						if err != nil {
							logger.Warning("%s: id=%d erreur lors de l'application à dispo_blocs_ac_sites: %v", table, row.ID, err)
						} else {
							stats.SitesUpdated += affected
							logger.Debug("%s: id=%d -> %d lignes mises à jour dans dispo_blocs_ac_sites", table, row.ID, affected)
						}
					}

					applied = true
					stats.Updated++
					logger.Debug("%s: id=%d -> est_disponible=%d (ticket=%v, ongoing=%v)",
						table, row.ID, *status, ticketIdentifier, segmentIsOngoing)
				}
			} else {
				stats.WithoutTicket++
				logger.Debug("%s: id=%d aucun ticket trouvé (ongoing=%v)", table, row.ID, segmentIsOngoing)
			}
			if err := upsertTracking(db, table, row, ticketIdentifier, applied, endEffective, segmentIsOngoing); err != nil {
				return stats, err
			}

			stats.Tracked++
		}
	}

	return stats, nil
}

func quoteIdentifier(name string) string {
	return "`" + name + "`"
}

func worker(id int, jobs <-chan TableJob, results chan<- Stats, db *sql.DB, mode string, start, end *time.Time, now time.Time, wg *sync.WaitGroup) {
	defer wg.Done()

	for job := range jobs {
		stats, err := processTable(db, job.Table, mode, start, end, now)
		if err != nil {
			logger.Error("Worker %d: erreur lors du traitement de %s: %v", id, job.Table, err)
			continue
		}
		results <- stats
	}
}

func main() {
	var (
		startDate     = flag.String("start-date", "", "Début de la période (format: 2006-01-02 15:04:05)")
		endDate       = flag.String("end-date", "", "Fin de la période (format: 2006-01-02 15:04:05)")
		fullReprocess = flag.Bool("full-reprocess", false, "Vide la table tracking et retraite tout")
		tablesFlag    = flag.String("tables", "", "Liste de tables séparées par des virgules")
		verbose       = flag.Bool("verbose", false, "Active les logs détaillés")
	)

	flag.Parse()

	logger = &Logger{verbose: *verbose}
	log.SetFlags(log.LstdFlags)

	var mode string
	if *fullReprocess {
		mode = "full"
	} else if *startDate != "" || *endDate != "" {
		if *startDate == "" || *endDate == "" {
			logger.Error("Les deux dates doivent être fournies pour le mode période")
			os.Exit(2)
		}
		mode = "period"
	} else {
		mode = "incremental"
	}

	var start, end *time.Time
	if mode == "period" {
		s, err := parseDate(*startDate)
		if err != nil {
			logger.Error("Date de début invalide: %v", err)
			os.Exit(2)
		}
		start = &s

		e, err := parseDate(*endDate)
		if err != nil {
			logger.Error("Date de fin invalide: %v", err)
			os.Exit(2)
		}
		end = &e
	}

	config := getDbConfig()
	logger.Info("Connexion à %s@%s:%s/%s (mode=%s)", config.User, config.Host, config.Port, config.Database, mode)

	db, err := createDbConnection(config)
	if err != nil {
		logger.Error("Erreur de connexion: %v", err)
		os.Exit(1)
	}
	defer db.Close()

	var tables []string
	if *tablesFlag != "" {
		for _, t := range regexp.MustCompile(`\s*,\s*`).Split(*tablesFlag, -1) {
			if t != "" {
				tables = append(tables, t)
			}
		}
	} else {
		tables, err = listPdcTables(db, config.Database)
		if err != nil {
			logger.Error("Erreur lors de la liste des tables: %v", err)
			os.Exit(1)
		}
	}

	if len(tables) == 0 {
		logger.Warning("Aucune table PDC trouvée")
		os.Exit(0)
	}

	now := time.Now().UTC()

	if mode == "full" {
		logger.Info("Vidage de la table exclusion_tracking")
		if _, err := db.Exec("TRUNCATE TABLE exclusion_tracking"); err != nil {
			logger.Error("Erreur lors du vidage de exclusion_tracking: %v", err)
			os.Exit(1)
		}
	}

	jobs := make(chan TableJob, len(tables))
	results := make(chan Stats, len(tables))
	var wg sync.WaitGroup

	for i := 1; i <= NumWorkers; i++ {
		wg.Add(1)
		go worker(i, jobs, results, db, mode, start, end, now, &wg)
	}

	for _, table := range tables {
		jobs <- TableJob{Table: table}
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	var totalStats Stats
	for stats := range results {
		totalStats.Updated += stats.Updated
		totalStats.Tracked += stats.Tracked
		totalStats.WithoutTicket += stats.WithoutTicket
		totalStats.SkippedBorne += stats.SkippedBorne
		totalStats.Split += stats.Split
		totalStats.Deleted += stats.Deleted
		totalStats.SitesUpdated += stats.SitesUpdated
	}

	logger.Info("Traitement terminé: %d lignes mises à jour, %d suivies, %d sans ticket, %d ignorées (borne_id), %d splittées, %d supprimées, %d sites mis à jour",
		totalStats.Updated, totalStats.Tracked, totalStats.WithoutTicket, totalStats.SkippedBorne, totalStats.Split, totalStats.Deleted, totalStats.SitesUpdated)
}
