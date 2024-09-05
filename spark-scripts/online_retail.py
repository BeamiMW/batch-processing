import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, max, lit, datediff, when, avg

def get_postgres_config():
    """Read PostgreSQL configuration from environment variables."""
    return {
        'user': os.environ.get('POSTGRES_USER', 'postgres'),
        'password': os.environ.get('POSTGRES_PASSWORD', 'admin'),
        'db': os.environ.get('POSTGRES_DW_DB', 'warehouse'),
        'port': '5432',
        'container_name': os.environ.get('POSTGRES_CONTAINER_NAME', 'demo-postgres')
    }

def create_spark_session():
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName("Retail Data Analysis") \
        .getOrCreate()

def load_data_from_postgres(spark, config):
    """Load data from PostgreSQL database into a Spark DataFrame."""
    jdbc_url = f"jdbc:postgresql://{config['container_name']}:{config['port']}/{config['db']}"
    return spark.read \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", "public.retail") \
        .option("user", config['user']) \
        .option("password", config['password']) \
        .option("driver", "org.postgresql.Driver") \
        .load()

def calculate_rfm(df):
    """Calculate Recency, Frequency, and Monetary (RFM) values."""
    df = df.withColumn("InvoiceDate", col("InvoiceDate").cast("timestamp"))
    most_recent_date = df.agg(max("InvoiceDate")).collect()[0][0]
    
    rfm = df.groupBy("CustomerID").agg(
        max("InvoiceDate").alias("LastPurchaseDate"),
        count("InvoiceNo").alias("Frequency"),
        sum(col("Quantity") * col("UnitPrice")).alias("Monetary")
    )
    
    rfm = rfm.withColumn("Recency", datediff(lit(most_recent_date), col("LastPurchaseDate")))
    return rfm

def identify_churn(rfm, churn_threshold=180):
    """Identify churned customers based on the Recency value."""
    return rfm.withColumn("Churn", when(col("Recency") > churn_threshold, 1).otherwise(0))

def calculate_churn_rate(rfm):
    """Calculate and return the churn rate."""
    total_customers = rfm.count()
    churned_customers = rfm.filter(col("Churn") == 1).count()
    churn_rate = churned_customers / total_customers
    return churn_rate, total_customers, churned_customers

def analyze_churn_patterns(rfm):
    """Analyze churn patterns and return summary statistics."""
    return rfm.groupBy("Churn").agg(
        avg("Recency").alias("AverageRecency"),
        avg("Frequency").alias("AverageFrequency"),
        avg("Monetary").alias("AverageMonetary")
    )

def calculate_avg_order_value(df):
    """Calculate and return the average order value."""
    return df.groupBy("InvoiceNo").agg(
        sum(col("Quantity") * col("UnitPrice")).alias("OrderValue")
    ).agg(avg("OrderValue").alias("AverageOrderValue"))

def identify_top_customers(df, limit=10):
    """Identify and return the top N frequent customers."""
    return df.groupBy("CustomerID").agg(
        count("InvoiceNo").alias("TotalOrders"),
        sum(col("Quantity") * col("UnitPrice")).alias("TotalSpent")
    ).orderBy(col("TotalOrders").desc()).limit(limit)

def save_to_postgres(df, analysis, config):
    """Save a Spark DataFrame to PostgreSQL."""
    jdbc_url = f"jdbc:postgresql://{config['container_name']}:{config['port']}/{config['db']}"
    df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", analysis) \
        .option("user", config['user']) \
        .option("password", config['password']) \
        .option("driver", "org.postgresql.Driver") \
        .mode("overwrite") \
        .save()

def save_to_csv(df, file_path):
    """Convert a Spark DataFrame to Pandas and save it as a CSV file."""
    df.toPandas().to_csv(file_path, index=False)

def main():
    config = get_postgres_config()
    spark = create_spark_session()
    
    df = load_data_from_postgres(spark, config)
    
    rfm = calculate_rfm(df)
    rfm = identify_churn(rfm)
    
    churn_rate, total_customers, churned_customers = calculate_churn_rate(rfm)
    print(f"Total Customers: {total_customers}")
    print(f"Churned Customers: {churned_customers}")
    print(f"Churn Rate: {churn_rate:.2%}")
    
    rfm_summary = analyze_churn_patterns(rfm)
    avg_order_value = calculate_avg_order_value(df)
    top_customers = identify_top_customers(df)
    
    print("RFM Summary:")
    rfm_summary.show()
    
    print("Average Order Value:")
    avg_order_value.show()
    
    print("Top 10 Frequent Customers:")
    top_customers.show()
    
    save_to_postgres(rfm, "rfm_analysis", config)
    save_to_postgres(rfm_summary, "rfm_summary", config)
    save_to_postgres(avg_order_value, "avg_order_value", config)
    save_to_postgres(top_customers, "top_customers", config)
    
    spark.stop()

if __name__ == "__main__":
    main()