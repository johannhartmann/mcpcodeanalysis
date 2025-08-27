# Docker Setup for MCP Code Analysis Server

This directory contains Docker-related configuration files for the MCP Code Analysis Server.

## Files Overview

- **entrypoint.sh** - Main entrypoint script that handles different service modes
- **init-db.sql** - PostgreSQL initialization script with pgvector extension
- **nginx.conf** - Nginx reverse proxy configuration
- **backup.sh** - Database backup script
- **prometheus.yml** - Prometheus monitoring configuration

## Quick Start

### Development Environment

```bash
# Start all services
docker-compose up -d

# Start only core services (postgres + mcp-server)
docker-compose up -d postgres mcp-server

# Run initial code scan
docker-compose run --rm scanner

# View logs
docker-compose logs -f mcp-server

# Access services
# - MCP Server: http://localhost:8080
# - PostgreSQL: localhost:5432
# - PgAdmin: http://localhost:5050 (profile: dev)
```

### Production-Ready Environment

The `docker-compose.prod.yml` file provides a production-ready configuration example with:
- Nginx reverse proxy
- Redis caching
- Resource limits
- Health checks
- Monitoring stack (optional)

```bash
# Start production-ready stack
docker-compose -f docker-compose.prod.yml up -d

# With monitoring
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# Run backup
docker-compose -f docker-compose.prod.yml run --rm backup
```

Note: Adjust the configuration according to your specific deployment environment.

## Service Profiles

### Development (docker-compose.yml)
- **Default**: postgres, mcp-server
- **dev**: Adds pgadmin for database management
- **tools**: Adds scanner and indexer services
- **cache**: Adds Redis for caching

### Production (docker-compose.prod.yml)
- **Default**: postgres, mcp-server, nginx, redis
- **monitoring**: Adds Prometheus and Grafana
- **backup**: Database backup service

## Environment Variables

Create a `.env` file with:

```bash
# Required
OPENAI_API_KEY=your-api-key
POSTGRES_PASSWORD=secure-password

# Optional
MCP_LOGGING__LEVEL=INFO
DEBUG=false
API_KEY=your-mcp-api-key
REDIS_PASSWORD=redis-password
GRAFANA_PASSWORD=admin-password
```

## Database Management

### Backup
```bash
# Manual backup
docker-compose -f docker-compose.prod.yml run --rm backup

# Scheduled backup (add to crontab)
0 2 * * * cd /path/to/project && docker-compose -f docker-compose.prod.yml run --rm backup
```

### Restore
```bash
# Restore from backup
docker exec -i mcp-postgres psql -U codeanalyzer code_analysis < backup.sql
```

### Migrations
Migrations are automatically run on server startup. To run manually:
```bash
docker-compose exec mcp-server python -m alembic upgrade head
```

## Monitoring

When using the monitoring profile:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/GRAFANA_PASSWORD)

## SSL Configuration

For production with HTTPS:

1. Place SSL certificates in `docker/ssl/`
2. Uncomment HTTPS section in `nginx.conf`
3. Update server_name with your domain

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs mcp-server

# Check PostgreSQL connection
docker-compose exec mcp-server pg_isready -h postgres
```

### Database issues
```bash
# Connect to database
docker-compose exec postgres psql -U codeanalyzer code_analysis

# Check pgvector extension
\dx
```

### Performance issues
```bash
# Check resource usage
docker stats

# Adjust limits in docker-compose.prod.yml
```
