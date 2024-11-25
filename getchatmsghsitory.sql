CREATE DATABASE chat_history;
USE chat_history;
CREATE TABLE chattingmsg(
user_id VARCHAR(255) NOT NULL,
userinput VARCHAR(255) NOT NULL,
aiout VARCHAR(255) NOT NULL,
CONSTRAINT chattingmsg_PK PRIMARY KEY(user_id)
);

DESC chattingmsg;
