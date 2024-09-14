import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, max, lit, datediff, when, avg
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
dotenv_path = Path("/opt/app/.env")
load_dotenv(dotenv_path=dotenv_path)

# Retrieve PostgreSQL configuration from environment variables.
def get_postgres_config():
    return {
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'db': os.getenv('POSTGRES_DW_DB'),
        'port': os.getenv('POSTGRES_PORT'),
        'container_name': os.getenv('POSTGRES_CONTAINER_NAME')
    }

# Create and return a Spark session.
def create_spark_session(app_name="Retail Data Analysis"):
    return SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

# Load data from PostgreSQL database into a Spark DataFrame.
def load_data_from_postgres(spark, config):
    jdbc_url = f"jdbc:postgresql://{config['container_name']}:{config['port']}/{config['db']}"
    return spark.read \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", "public.retail") \
        .option("user", config['user']) \
        .option("password", config['password']) \
        .option("driver", "org.postgresql.Driver") \
        .load()

# Calculate Recency, Frequency, and Monetary (RFM) values.
def calculate_rfm(df):
    df = df.withColumn("InvoiceDate", col("InvoiceDate").cast("timestamp"))
    most_recent_date = df.agg(max("InvoiceDate")).collect()[0][0]
    
    rfm = df.groupBy("CustomerID").agg(
        max("InvoiceDate").alias("LastPurchaseDate"),
        count("InvoiceNo").alias("Frequency"),
        sum(col("Quantity") * col("UnitPrice")).alias("Monetary")
    )
    
    rfm = rfm.withColumn("Recency", datediff(lit(most_recent_date), col("LastPurchaseDate")))
    return rfm


# Identify churned customers based on the Recency value.
def identify_churn(rfm, churn_threshold=180):
    return rfm.withColumn("Churn", when(col("Recency") > churn_threshold, 1).otherwise(0))

# Calculate and return the churn rate.
def calculate_churn_rate(rfm):
    total_customers = rfm.count()
    churned_customers = rfm.filter(col("Churn") == 1).count()
    churn_rate = churned_customers / total_customers
    return churn_rate, total_customers, churned_customers

# Analyze churn patterns and return summary statistics.
def analyze_churn_patterns(rfm):
    return rfm.groupBy("Churn").agg(
        avg("Recency").alias("AverageRecency"),
        avg("Frequency").alias("AverageFrequency"),
        avg("Monetary").alias("AverageMonetary")
    )

# Calculate and return the average order value.
def calculate_avg_order_value(df):
    return df.groupBy("InvoiceNo").agg(
        sum(col("Quantity") * col("UnitPrice")).alias("OrderValue")
    ).agg(avg("OrderValue").alias("AverageOrderValue"))

#  Identify and return the top N frequent customers.
def identify_top_customers(df, limit=10):
    return df.groupBy("CustomerID").agg(
        count("InvoiceNo").alias("TotalOrders"),
        sum(col("Quantity") * col("UnitPrice")).alias("TotalSpent")
    ).orderBy(col("TotalOrders").desc()).limit(limit)

#Save a Spark DataFrame to PostgreSQL.
def save_to_postgres(df, table_name, config):
    jdbc_url = f"jdbc:postgresql://{config['container_name']}:{config['port']}/{config['db']}"
    df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", table_name) \
        .option("user", config['user']) \
        .option("password", config['password']) \
        .option("driver", "org.postgresql.Driver") \
        .mode("overwrite") \
        .save()

# Save a Spark DataFrame to CSV.
def save_to_csv(df, file_path):
    df.coalesce(1).write.mode("overwrite").csv(file_path, header=True)

def main():
    # Load configurations and Spark session
    config = get_postgres_config()
    spark = create_spark_session()

    # Load data from PostgreSQL
    df = load_data_from_postgres(spark, config)

    # Calculate RFM values and identify churn
    rfm = calculate_rfm(df)
    rfm = identify_churn(rfm)

    # Calculate churn rate
    churn_rate, total_customers, churned_customers = calculate_churn_rate(rfm)
    print(f"Total Customers: {total_customers}")
    print(f"Churned Customers: {churned_customers}")
    print(f"Churn Rate: {churn_rate:.2%}")

    # Analyze churn patterns, average order value, and top customers
    rfm_summary = analyze_churn_patterns(rfm)
    avg_order_value = calculate_avg_order_value(df)
    top_customers = identify_top_customers(df)

    # Show summaries
    print("RFM Summary:")
    rfm_summary.show()

    print("Average Order Value:")
    avg_order_value.show()

    print("Top 10 Frequent Customers:")
    top_customers.show()

    # Save results back to PostgreSQL
    save_to_postgres(rfm, "rfm_analysis", config)
    save_to_postgres(rfm_summary, "rfm_summary", config)
    save_to_postgres(avg_order_value, "avg_order_value", config)
    save_to_postgres(top_customers, "top_customers", config)

    # Save results to CSV
    save_to_csv(rfm, "data/rfm_analysis.csv") 
    save_to_csv(rfm_summary, "data/rfm_summary.csv")  
    save_to_csv(avg_order_value, "data/avg_order_value.csv") 
    save_to_csv(top_customers, "data/top_customers.csv")  

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
