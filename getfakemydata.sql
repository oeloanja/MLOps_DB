CREATE DATABASE mydata;
USE mydata;
CREATE TABLE user_log(
user_id VARCHAR(255) NOT NULL,
int_rate FLOAT NOT NULL,
installment FLOAT NOT NULL,
issue_d_period VARCHAR(20) NOT NULL,
debt INT NOT NULL,
cr_line_period VARCHAR(255) NOT NULL,
pub_rec INT NOT NULL,
revol_bal INT NOT NULL,
revol_util FLOAT NOT NULL,
total_acc INT NOT NULL,
mort_acc INT NOT NULL,
collections_12_mths_ex_med FLOAT NOT NULL,
CONSTRAINT user_log_PK PRIMARY KEY(user_id)
);

DESC user_log;