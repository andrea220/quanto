"""
Configuration settings for Interactive Brokers connection.
"""

# Default configuration for IB Gateway/TWS connection
IB_CONFIG = {
    # Connection settings
    'host': '127.0.0.1',
    'port': 4001,  # 7497 for TWS paper, 4001 for IB Gateway paper, 7496 for TWS live
    'client_id': 1,  # Unique client ID (can be 0-32)
    
    # Timeout settings
    'connect_timeout': 10,  # seconds
    'request_timeout': 30,  # seconds
    
    # Data settings
    'default_exchange': 'SMART',  # Default exchange for stocks
    'default_currency': 'USD',
    
    # Rate limiting
    'max_requests_per_interval': 60,  # Max historical data requests
    'rate_limit_interval': 600,  # 10 minutes in seconds
}


def get_config():
    """
    Get the IB configuration.
    
    Returns:
        dict: Configuration dictionary
    """
    return IB_CONFIG.copy()


def update_config(**kwargs):
    """
    Update configuration parameters.
    
    Args:
        **kwargs: Configuration parameters to update
    """
    IB_CONFIG.update(kwargs)
