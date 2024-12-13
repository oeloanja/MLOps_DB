CREATE DATABASE mydata;
USE mydata;
CREATE TABLE user_log(
user_pn VARCHAR(255) NOT NULL,
int_rate FLOAT NOT NULL,
installment FLOAT NOT NULL,
issue_d_period FLOAT NOT NULL,
debt INT NOT NULL,
cr_line_period FLOAT NOT NULL,
pub_rec FLOAT NOT NULL,
revol_bal FLOAT NOT NULL,
revol_util FLOAT NOT NULL,
open_acc FLOAT NOT NULL,
total_acc FLOAT NOT NULL,
mort_acc FLOAT NOT NULL,
collections_12_mths_ex_med FLOAT NOT NULL,
mortgage_debt INTEGER NOT NULL,
mortgage_repayment INTEGER NOT NULL,
repayment INTEGER NOT NULL,
mortgage_term INTEGER NOT NULL,
CONSTRAINT user_log_PK PRIMARY KEY(user_pn)
);

DESC user_log;