USE mydata;
ALTER TABLE user_log ADD COLUMN mortgage_debt INTEGER NOT NULL;
ALTER TABLE user_log ADD COLUMN mortgage_repayment INTEGER NOT NULL;
ALTER TABLE user_log ADD COLUMN repayment INTEGER NOT NULL;
ALTER TABLE user_log ADD COLUMN mortgage_term INTEGER NOT NULL;