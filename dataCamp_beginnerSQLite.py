https://www.datacamp.com/tutorial/beginners-guide-to-sqlite

# Benefits of SQLite
    1. No Servers
    2. Simple Database Files 
    3. Manifest typing = any amount of any data type in any column
        *EXCEPT* primary keys, which must be integers

# Limitations of SQLite
    1. No Right or Full Outer Joins
    2. Security limitations due to not being on a server

# Installation 
    https://www.sqlite.org/download.html
    Download "sqlite-tools-win32-x86-3270200.zip"
    Unzip the file, see...
        sqldiff
        sqlite3
        sqlite3_analyzer
    run sqlite3 by typing 'sqlite3 <Enter>' in the command line

# Create a new database
    .open 'database_name.db' - creates a database with the selected name
    .mode csv - sets csv mode
    .headers on - displays the column headers
    .import 'your_csv_file.csv' table_name - imports a csv into the table name of your choice

    .tables - display all the available tables in the database file
    .schema table_name - display the table structure

    SELECT $$
    FROM table_name*        verifies that the data is there

# Creating and Adding to Tables
    CREATE TABLE table_name (       creates a table with the following characteristics
            "user_id" TEXT,
            "screen_name" TEXT,
            "tweets" TEXT,
            "likes" INT);

    INSERT INTO table_name (user_id, screen_name, tweet, likes)
    VALUES (1122567, 'Tutorial', 'SQLite Rules', 10);

# Exporting a csv from SQLite
    .header on
    .mode csv
    .output 'your_new_csv_file_name.csv'
    SELECT *    # which columns you want to export
    FROM target_table;  # from which table
    .exit   # exit the operation

# Updating an existing table
    UPDATE target_table
    SET column_you_want = 'your stuff you want to add'
    WHERE user_id = 'value in the primary key that signifies the row you want to edit';

# Deleting a table from the database
    DROP TABLE target_table;

# Counting the unique values in a particular column of your table
    SELECT COUNT (DISTINCT target_column)
    FROM target_table

    > returns the number

# Looking to see who has the most values in the tweet example
    SELECT screen_name, COUNT() AS total_tweets 
    FROM sample_tweets
    GROUP BY screen_name
    ORDER BY total_tweets DESC
    LIMIT 20; 

# Selecting the tweet content by a specific user
    SELECT text 
    FROM sample_tweets
    WHERE screen_name = "Uwemuegge"
    LIMIT 10;


