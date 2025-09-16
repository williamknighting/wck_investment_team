"""
DuckDB Market Data Manager
High-performance analytical database for market data storage
"""
import duckdb
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

try:
    from ..utils.logging_config import get_logger
except ImportError:
    from utils.logging_config import get_logger


class DuckDBDataManager:
    """
    High-performance market data storage using DuckDB
    Optimized for analytical queries and fast aggregations
    """
    
    def __init__(self, db_path: str = "data_store/market_data.duckdb"):
        """
        Initialize DuckDB data manager
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.logger = get_logger("duckdb_manager")
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self.conn = duckdb.connect(str(self.db_path))
        
        # Initialize schema
        self._init_schema()
        
        self.logger.info(f"DuckDB Manager initialized: {db_path}")
    
    def _init_schema(self):
        """Initialize DuckDB schema with optimized tables"""
        
        # Main market data table with columnar optimization
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                interval VARCHAR NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume BIGINT,
                adj_close DOUBLE PRECISION,
                vwap DOUBLE PRECISION,
                trade_count INTEGER,
                data_source VARCHAR DEFAULT 'alpaca',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (symbol, timestamp, interval)
            )
        """)
        
        # Create indexes for fast queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_interval ON market_data (symbol, interval)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data (timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON market_data (symbol, timestamp)")
        
        # Metadata table for tracking data freshness
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_metadata (
                symbol VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                data_source VARCHAR NOT NULL,
                last_updated TIMESTAMPTZ NOT NULL,
                record_count INTEGER NOT NULL,
                oldest_record TIMESTAMPTZ,
                newest_record TIMESTAMPTZ,
                PRIMARY KEY (symbol, interval, data_source)
            )
        """)
        
        # Performance metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_stats (
                table_name VARCHAR,
                record_count BIGINT,
                size_mb DOUBLE PRECISION,
                last_vacuum TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        self.logger.info("DuckDB schema initialized")
    
    def store_market_data(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        interval: str,
        data_source: str = "alpaca"
    ) -> bool:
        """
        Store market data in DuckDB with high performance
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            interval: Data interval (1d, 1h, 5m, etc.)
            data_source: Data source name
            
        Returns:
            Success status
        """
        try:
            if data.empty:
                self.logger.warning(f"Empty data provided for {symbol}")
                return False
            
            # SAFETY CHECK: Never allow mock data in production DuckDB
            if data_source == "mock":
                self.logger.error(f"REJECTED: Mock data attempted to be stored in DuckDB for {symbol}. DuckDB is for real data only!")
                return False
            
            # Prepare data for storage
            df = data.copy()
            
            # Handle timestamp column/index
            if 'timestamp' not in df.columns:
                # If no timestamp column, create one from index
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    df = df.rename(columns={'index': 'timestamp'})
                else:
                    # Try to convert index to datetime
                    df.index = pd.to_datetime(df.index)
                    df = df.reset_index()
                    df = df.rename(columns={'index': 'timestamp'})
            else:
                # Timestamp column already exists, ensure it's datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add required columns
            df['symbol'] = symbol
            df['interval'] = interval
            df['data_source'] = data_source
            df['created_at'] = datetime.now(timezone.utc)
            df['updated_at'] = datetime.now(timezone.utc)
            
            # Rename columns to match schema (standardize common variations)
            column_mapping = {
                'Date': 'timestamp',
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low', 
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close',
                'VWAP': 'vwap',
                'Trade Count': 'trade_count'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['symbol', 'timestamp', 'interval', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Missing required column {col} for {symbol}")
                    return False
            
            # Fill optional columns with defaults
            if 'adj_close' not in df.columns:
                df['adj_close'] = df['close']
            if 'vwap' not in df.columns:
                df['vwap'] = None
            if 'trade_count' not in df.columns:
                df['trade_count'] = None
            
            # Remove existing data for this symbol/interval (upsert behavior)
            self.conn.execute("""
                DELETE FROM market_data 
                WHERE symbol = ? AND interval = ? AND data_source = ?
            """, [symbol, interval, data_source])
            
            # Insert new data using DuckDB's efficient bulk insert
            df_clean = df[[
                'symbol', 'timestamp', 'interval', 'open', 'high', 'low', 
                'close', 'volume', 'adj_close', 'vwap', 'trade_count',
                'data_source', 'created_at', 'updated_at'
            ]]
            
            self.conn.register('temp_df', df_clean)
            self.conn.execute("""
                INSERT INTO market_data SELECT * FROM temp_df
            """)
            self.conn.unregister('temp_df')
            
            # Update metadata
            self._update_metadata(symbol, interval, data_source, len(df))
            
            self.logger.info(f"Stored {len(df)} records for {symbol} {interval} from {data_source}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing data for {symbol}: {e}")
            return False
    
    def get_market_data(
        self, 
        symbol: str, 
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve market data with high-performance queries
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records
            
        Returns:
            DataFrame with market data
        """
        try:
            # Build query with filters
            where_conditions = ["symbol = ?", "interval = ?"]
            params = [symbol, interval]
            
            if start_date:
                where_conditions.append("timestamp >= ?")
                params.append(start_date)
            
            if end_date:
                where_conditions.append("timestamp <= ?")
                params.append(end_date)
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT timestamp, open, high, low, close, volume, adj_close, vwap, trade_count
                FROM market_data 
                WHERE {where_clause}
                ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query and return DataFrame
            df = self.conn.execute(query, params).fetchdf()
            
            if not df.empty:
                # Set timestamp as index
                df = df.set_index('timestamp')
                
                # Ensure proper column names (title case for compatibility)
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'adj_close': 'Adj Close',
                    'vwap': 'VWAP',
                    'trade_count': 'Trade Count'
                })
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_symbols(self, interval: str = None) -> List[str]:
        """Get list of available symbols"""
        try:
            query = "SELECT DISTINCT symbol FROM market_data"
            params = []
            
            if interval:
                query += " WHERE interval = ?"
                params.append(interval)
                
            query += " ORDER BY symbol"
            
            result = self.conn.execute(query, params).fetchall()
            return [row[0] for row in result]
            
        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return []
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of stored data"""
        try:
            summary = self.conn.execute("""
                SELECT 
                    symbol,
                    interval,
                    data_source,
                    COUNT(*) as record_count,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM market_data 
                GROUP BY symbol, interval, data_source
                ORDER BY symbol, interval
            """).fetchdf()
            
            total_records = self.conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
            
            return {
                "total_records": total_records,
                "summary_by_symbol": summary.to_dict('records'),
                "unique_symbols": len(summary['symbol'].unique()),
                "data_sources": summary['data_source'].unique().tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {"error": str(e)}
    
    def _update_metadata(self, symbol: str, interval: str, data_source: str, record_count: int):
        """Update metadata table with data statistics"""
        try:
            # Get date range
            result = self.conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM market_data 
                WHERE symbol = ? AND interval = ? AND data_source = ?
            """, [symbol, interval, data_source]).fetchone()
            
            oldest_record, newest_record = result
            
            # Upsert metadata
            self.conn.execute("""
                INSERT INTO data_metadata (symbol, interval, data_source, last_updated, record_count, oldest_record, newest_record)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, interval, data_source) DO UPDATE SET
                    last_updated = EXCLUDED.last_updated,
                    record_count = EXCLUDED.record_count,
                    oldest_record = EXCLUDED.oldest_record,
                    newest_record = EXCLUDED.newest_record
            """, [symbol, interval, data_source, datetime.now(timezone.utc), 
                  record_count, oldest_record, newest_record])
            
        except Exception as e:
            self.logger.error(f"Error updating metadata: {e}")
    
    def vacuum_and_analyze(self):
        """Optimize database performance"""
        try:
            self.conn.execute("VACUUM")
            self.conn.execute("ANALYZE")
            self.logger.info("Database vacuum and analyze completed")
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
            self.logger.info("DuckDB connection closed")


# Global instance
duckdb_manager: Optional[DuckDBDataManager] = None


def get_duckdb_manager() -> DuckDBDataManager:
    """Get global DuckDB manager instance"""
    global duckdb_manager
    if duckdb_manager is None:
        duckdb_manager = DuckDBDataManager()
    return duckdb_manager