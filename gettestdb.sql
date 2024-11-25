CREATE DATABASE testdb;
use testdb;
CREATE TABLE borrow_user(
user_id VARCHAR(255) NOT NULL,
username VARCHAR(20) NOT NULL,
email VARCHAR(255) NOT NULL,
CONSTRAINT borrow_user_PK PRIMARY KEY(user_id)
);
INSERT INTO borrow_user (user_id, username, email) VALUES ('eng1001', 'engunyeon', 'eng1001@gmail.com');
INSERT INTO borrow_user (user_id, username, email) VALUES ('sawadi777', 'sawadi', 'sa3009@gmail.com');
INSERT INTO borrow_user (user_id, username, email) VALUES ('wang4432', 'wang', 'wang4321@hwawei.com');
INSERT INTO borrow_user (user_id, username, email) VALUES ('drake321', 'drake', 'drake4321@gmail.com');
INSERT INTO borrow_user (user_id, username, email) VALUES ('lucky777', 'lucky', 'lucky777@gmail.com');

DESC borrow_user;

# 다 가상의 데이터임
# 이걸 굳이 배껴서 쓴다고?

