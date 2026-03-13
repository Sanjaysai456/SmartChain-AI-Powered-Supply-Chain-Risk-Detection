    CREATE TABLE supply_chain_data (
        temperature FLOAT,
        humidity FLOAT,
        vibration_level FLOAT,
        delivery_delay_days INT,
        system_alert_count INT,
        manual_risk_label VARCHAR(20)
    );

    LOAD DATA INFILE 'data.csv'
    INTO TABLE supply_chain_data
    FIELDS TERMINATED BY ','
    IGNORE 1 ROWS;


    -- Risk Distribution
    SELECT 
        manual_risk_label,
        COUNT(*) AS total_cases
    FROM supply_chain_data
    GROUP BY manual_risk_label;

    SELECT 
        manual_risk_label,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM supply_chain_data) AS percentage
    FROM supply_chain_data
    GROUP BY manual_risk_label;


    SELECT
        manual_risk_label,
        AVG(temperature) AS avg_temperature,
        AVG(humidity) AS avg_humidity,
        AVG(vibration_level) AS avg_vibration
    FROM supply_chain_data
    GROUP BY manual_risk_label;


    SELECT
        manual_risk_label,
        AVG(delivery_delay_days) AS avg_delay
    FROM supply_chain_data
    GROUP BY manual_risk_label;


    SELECT
        AVG(vibration_level) AS avg_vibration,
        AVG(delivery_delay_days) AS avg_delay,
        AVG(system_alert_count) AS avg_alerts
    FROM supply_chain_data
    WHERE manual_risk_label = 'High Risk';


    SELECT *
    FROM supply_chain_data
    ORDER BY vibration_level DESC
    LIMIT 10;