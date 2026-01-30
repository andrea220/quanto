"""
Connection manager for Interactive Brokers API.
"""

import logging
from typing import Optional
from ib_insync import IB, util
from config.ib_config import get_config


# Setup logging
logger = logging.getLogger(__name__)


class IBConnection:
    """
    Connection manager for Interactive Brokers TWS/Gateway.
    
    This class manages the connection to IB and provides a context manager
    interface for automatic cleanup.
    
    Example:
        >>> with IBConnection() as ib:
        ...     # Use ib connection
        ...     contract = Stock('AAPL', 'SMART', 'USD')
        ...     ib.qualifyContracts(contract)
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        connect_timeout: Optional[int] = None,
        readonly: bool = False
    ):
        """
        Initialize the IB connection manager.
        
        Args:
            host: IB Gateway/TWS host address. Defaults to config value.
            port: IB Gateway/TWS port. Defaults to config value.
            client_id: Unique client ID. Defaults to config value.
            connect_timeout: Connection timeout in seconds. Defaults to config value.
            readonly: If True, connect in read-only mode.
        """
        self.config = get_config()
        
        self.host = host or self.config['host']
        self.port = port or self.config['port']
        self.client_id = client_id or self.config['client_id']
        self.connect_timeout = connect_timeout or self.config['connect_timeout']
        self.readonly = readonly
        
        self._ib: Optional[IB] = None
        self._connected = False
        
        logger.info(f"IBConnection initialized - Host: {self.host}, Port: {self.port}, ClientID: {self.client_id}")
    
    def connect(self) -> IB:
        """
        Establish connection to IB Gateway/TWS.
        
        Returns:
            IB: Connected IB instance
            
        Raises:
            ConnectionError: If connection fails
        """
        if self._connected and self._ib and self._ib.isConnected():
            logger.info("Already connected to IB")
            return self._ib
        
        try:
            self._ib = IB()
            
            # Enable event loop integration
            util.startLoop()
            
            logger.info(f"Connecting to IB at {self.host}:{self.port} with client ID {self.client_id}...")
            self._ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.connect_timeout,
                readonly=self.readonly
            )
            
            self._connected = True
            logger.info("Successfully connected to Interactive Brokers")
            
            return self._ib
            
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            raise ConnectionError(f"Failed to connect to Interactive Brokers: {e}")
    
    def disconnect(self):
        """
        Disconnect from IB Gateway/TWS.
        """
        if self._ib and self._connected:
            try:
                logger.info("Disconnecting from IB...")
                self._ib.disconnect()
                self._connected = False
                logger.info("Successfully disconnected from IB")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
    
    def is_connected(self) -> bool:
        """
        Check if currently connected to IB.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self._connected and self._ib is not None and self._ib.isConnected()
    
    def get_ib(self) -> IB:
        """
        Get the IB instance. Connects if not already connected.
        
        Returns:
            IB: The IB instance
            
        Raises:
            ConnectionError: If not connected and connection fails
        """
        if not self.is_connected():
            return self.connect()
        return self._ib
    
    def __enter__(self):
        """
        Context manager entry - establishes connection.
        
        Returns:
            IB: Connected IB instance
        """
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - disconnects and cleans up.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.disconnect()
        return False
    
    def __del__(self):
        """
        Destructor - ensures disconnection on object deletion.
        """
        self.disconnect()

