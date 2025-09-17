# Scripts

Utility scripts for system operation and agent tools.

## Available Scripts

### Data Management
- **`backfill_historical_data.py`** - Initial data setup, pulls 2 years of historical data
- **`simple_data_refresh.py`** - Daily data refresh, used by Data Refresh Agent

## Usage

These scripts can be used by:
1. **Direct execution** - Run manually for setup/maintenance
2. **Agent integration** - Research agents can call these for data operations
3. **System automation** - Scheduled data refresh operations

## Agent Integration

Research agents and data agents can import and use these scripts:

```python
from scripts.simple_data_refresh import SimpleDataRefresh
from scripts.backfill_historical_data import HistoricalDataBackfill
```