-- vim: set ft=sql :
--
-- psql -h localhost -U m.anselmi -w -d biomappings -f init-biomappings-db-postgres.sql

-- DROP DATABASE IF EXISTS biomappings;
-- CREATE DATABASE biomappings;

DROP SCHEMA IF EXISTS resources;
CREATE SCHEMA resources;

SET search_path TO resources,"$user",public;

DROP TABLE IF EXISTS curators;
CREATE TABLE curators (
    "user" TEXT,
    orcid TEXT NOT NULL,
    name TEXT,
    PRIMARY KEY (orcid)
);

DROP TABLE IF EXISTS incorrect;
CREATE TABLE incorrect (
    source_prefix TEXT,
    source_identifier TEXT,
    source_name TEXT,
    relation TEXT,
    target_prefix TEXT,
    target_identifier TEXT,
    target_name TEXT,
    type TEXT,
    source TEXT,
    prediction_type TEXT,
    prediction_source TEXT,
    prediction_confidence DOUBLE PRECISION
);

DROP TABLE IF EXISTS mappings;
CREATE TABLE mappings (
    source_prefix TEXT,
    source_identifier TEXT,
    source_name TEXT,
    relation TEXT,
    target_prefix TEXT,
    target_identifier TEXT,
    target_name TEXT,
    type TEXT,
    source TEXT,
    prediction_type TEXT,
    prediction_source TEXT,
    prediction_confidence DOUBLE PRECISION
);

DROP TABLE IF EXISTS predictions;
CREATE TABLE predictions (
    source_prefix TEXT,
    source_identifier TEXT,
    source_name TEXT,
    relation TEXT,
    target_prefix TEXT,
    target_identifier TEXT,
    target_name TEXT,
    type TEXT,
    confidence DOUBLE PRECISION,
    source TEXT
);

DROP TABLE IF EXISTS unsure;
CREATE TABLE unsure (
    source_prefix TEXT,
    source_identifier TEXT,
    source_name TEXT,
    relation TEXT,
    target_prefix TEXT,
    target_identifier TEXT,
    target_name TEXT,
    type TEXT,
    source TEXT,
    prediction_type TEXT,
    prediction_source TEXT,
    prediction_confidence DOUBLE PRECISION
);
