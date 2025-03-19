-- vim: set ft=sql :
--
-- rm -f -- biomappings.duckdb && duckdb -f create-biomappings-duckdb.sql

ATTACH 'biomappings.duckdb' AS biomappings_db (
    TYPE duckdb,
    STORAGE_VERSION 'v1.2.1'
);
USE biomappings_db;

DROP TABLE IF EXISTS curators;
CREATE TABLE curators (
    user TEXT NOT NULL,
    orcid TEXT NOT NULL,
    name TEXT NOT NULL
);

COPY curators
FROM '/Users/m.anselmi/repos/biomappings/src/biomappings/resources/curators.tsv' (
    FORMAT csv,
    DELIMITER '\t',
    HEADER
);

-- COPY curators TO 'curators.parquet' (
--     FORMAT parquet,
--     COMPRESSION zstd,
--     COMPRESSION_LEVEL 22,
--     FIELD_IDS auto
-- );

DROP TABLE IF EXISTS incorrect;
CREATE TABLE incorrect (
    source_prefix TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    source_name TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_prefix TEXT NOT NULL,
    target_identifier TEXT NOT NULL,
    target_name TEXT NOT NULL,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    prediction_type TEXT,
    prediction_source TEXT,
    prediction_confidence DOUBLE
);

COPY incorrect
FROM '/Users/m.anselmi/repos/biomappings/src/biomappings/resources/incorrect.tsv' (
    FORMAT csv,
    DELIMITER '\t',
    HEADER
);

-- COPY incorrect TO 'incorrect.parquet' (
--     FORMAT parquet,
--     COMPRESSION zstd,
--     COMPRESSION_LEVEL 22,
--     FIELD_IDS auto
-- );

DROP TABLE IF EXISTS mappings;
CREATE TABLE mappings (
    source_prefix TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    source_name TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_prefix TEXT NOT NULL,
    target_identifier TEXT NOT NULL,
    target_name TEXT,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    prediction_type TEXT,
    prediction_source TEXT,
    prediction_confidence DOUBLE
);

COPY mappings
FROM '/Users/m.anselmi/repos/biomappings/src/biomappings/resources/mappings.tsv' (
    FORMAT csv,
    DELIMITER '\t',
    HEADER
);

-- .mode line
-- SELECT * FROM mappings WHERE target_name IS NULL;

-- COPY mappings TO 'mappings.parquet' (
--     FORMAT parquet,
--     COMPRESSION zstd,
--     COMPRESSION_LEVEL 22,
--     FIELD_IDS auto
-- );

DROP TABLE IF EXISTS predictions;
CREATE TABLE predictions (
    source_prefix TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    source_name TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_prefix TEXT NOT NULL,
    target_identifier TEXT NOT NULL,
    target_name TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence DOUBLE NOT NULL,
    source TEXT NOT NULL
);

COPY predictions
FROM '/Users/m.anselmi/repos/biomappings/src/biomappings/resources/predictions.tsv' (
    FORMAT csv,
    DELIMITER '\t',
    HEADER
);

-- COPY predictions TO 'predictions.parquet' (
--     FORMAT parquet,
--     COMPRESSION zstd,
--     COMPRESSION_LEVEL 22,
--     FIELD_IDS auto
-- );

DROP TABLE IF EXISTS unsure;
CREATE TABLE unsure (
    source_prefix TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    source_name TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_prefix TEXT NOT NULL,
    target_identifier TEXT NOT NULL,
    target_name TEXT,
    type TEXT NOT NULL,
    source TEXT NOT NULL,
    prediction_type TEXT,
    prediction_source TEXT,
    prediction_confidence DOUBLE
);

COPY unsure
FROM '/Users/m.anselmi/repos/biomappings/src/biomappings/resources/unsure.tsv' (
    FORMAT csv,
    DELIMITER '\t',
    HEADER
);

-- .mode line
-- SELECT * FROM unsure WHERE target_name IS NULL;

-- COPY unsure TO 'unsure.parquet' (
--     FORMAT parquet,
--     COMPRESSION zstd,
--     COMPRESSION_LEVEL 22,
--     FIELD_IDS auto
-- );

ATTACH ':memory:' AS memory_db;
USE memory_db;
DETACH biomappings_db;
