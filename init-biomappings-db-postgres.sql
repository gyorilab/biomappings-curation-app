-- vim: set ft=sql :
--
-- psql -h localhost -U m.anselmi -w -d biomappings -f init-biomappings-db-postgres.sql

-- DROP DATABASE IF EXISTS biomappings;
-- CREATE DATABASE biomappings;

DROP SCHEMA IF EXISTS curation_interface;
CREATE SCHEMA curation_interface;

SET search_path TO curation_interface,"$user",public;

DROP DOMAIN IF EXISTS domain_user_id CASCADE;
CREATE DOMAIN domain_user_id AS TEXT CHECK (
  VALUE ~ 'orcid:[0-9]{4}(-[0-9]{4}){3}'
);

DROP DOMAIN IF EXISTS domain_non_negative_bigint CASCADE;
CREATE DOMAIN domain_non_negative_bigint AS BIGINT CHECK (
  VALUE >= 0
);

DROP DOMAIN IF EXISTS domain_mapping_classification CASCADE;
CREATE DOMAIN domain_mapping_classification AS TEXT CHECK (
  VALUE IN ('broad', 'correct', 'incorrect', 'narrow', 'unsure')
);

DROP TABLE IF EXISTS tbl_mark;
CREATE TABLE tbl_mark (
    user_id domain_user_id NOT NULL,
    line domain_non_negative_bigint NOT NULL,
    value domain_mapping_classification NOT NULL,
    PRIMARY KEY (user_id, line)
);
