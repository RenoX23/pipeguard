CREATE DATABASE pipeguard;
\c pipeguard;

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS dlq;
CREATE SCHEMA IF NOT EXISTS monitoring;

CREATE TABLE monitoring.structural_scores (
    id                SERIAL PRIMARY KEY,
    run_id            VARCHAR(50),
    dataset           VARCHAR(50),
    batch_date        DATE,
    structural_score  FLOAT,
    null_violations   INT,
    schema_violations INT,
    format_violations INT,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE TABLE monitoring.semantic_scores (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(50),
    dataset         VARCHAR(50),
    batch_date      DATE,
    if_score        FLOAT,
    ks_psi_score    FLOAT,
    adwin_score     FLOAT,
    composite_score FLOAT,
    tier            VARCHAR(20),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE monitoring.injection_log (
    id              SERIAL PRIMARY KEY,
    dataset         VARCHAR(50),
    injection_type  VARCHAR(50),
    injection_seed  INT,
    affected_rows   INT,
    run_id          VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW()
);
