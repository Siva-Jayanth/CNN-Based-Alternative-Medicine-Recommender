CREATE DATABASE alternative_medicine_db;
USE alternative_medicine_db;

CREATE TABLE admin (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50),
    PASSWORD VARCHAR(50)
);

INSERT INTO admin VALUES (1,'admin','admin');

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    NAME VARCHAR(100),
    email VARCHAR(100),
    PASSWORD VARCHAR(100)
);

CREATE TABLE predictions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    primary_symptom VARCHAR(100),
    severity VARCHAR(50),
    result VARCHAR(200)
);
