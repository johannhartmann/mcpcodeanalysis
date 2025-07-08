#!/bin/sh
set -e

# Configuration
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/mcp_backup_${TIMESTAMP}.sql.gz"
RETENTION_DAYS=7

# Ensure backup directory exists
mkdir -p "${BACKUP_DIR}"

echo "Starting backup at $(date)"

# Perform backup
pg_dump \
    -h postgres \
    -U codeanalyzer \
    -d code_analysis \
    --verbose \
    --no-owner \
    --no-privileges \
    --clean \
    --if-exists \
    | gzip > "${BACKUP_FILE}"

echo "Backup completed: ${BACKUP_FILE}"
echo "Backup size: $(du -h ${BACKUP_FILE} | cut -f1)"

# Clean up old backups
echo "Cleaning up backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -name "mcp_backup_*.sql.gz" -type f -mtime +${RETENTION_DAYS} -delete

echo "Backup process completed at $(date)"

# List remaining backups
echo "Current backups:"
ls -lh "${BACKUP_DIR}"/mcp_backup_*.sql.gz 2>/dev/null || echo "No backups found"