use role accountadmin;
create database if not exists outerbounds_demo;
create schema if not exists outerbounds_demo.tpch_predictions_schema;
grant all on schema outerbounds_demo.tpch_predictions_schema to role accountadmin;