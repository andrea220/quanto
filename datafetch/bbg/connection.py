"""
Connection manager for Bloomberg API using xbbg.
"""

import logging
from typing import Optional

# Setup logging
logger = logging.getLogger(__name__)


class BloombergConnection:
    """
    Connection manager for Bloomberg API using xbbg.
    
    Note: xbbg uses the Bloomberg Terminal API (BLP) which requires
    an active Bloomberg Terminal session. No explicit connection is needed
    as xbbg automatically uses the running Terminal session.
    
    Example:
        >>> connection = BloombergConnection()
        >>> # xbbg functions can be called directly
        >>> from xbbg import blp
        >>> df = blp.bdh('AAPL US Equity', ...)
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """
        Initialize the Bloomberg connection manager.
        
        Args:
            host: Not used (xbbg uses Terminal session)
            port: Not used (xbbg uses Terminal session)
        """
        self.host = host
        self.port = port
        self._connected = True  # xbbg is always "connected" if Terminal is running
        
        logger.info("BloombergConnection initialized (using xbbg with Terminal session)")
    
    def connect(self):
        """
        Establish connection to Bloomberg API.
        
        Note: With xbbg, no explicit connection is needed if Bloomberg Terminal
        is running. This method just verifies that xbbg can be imported.
        
        Raises:
            ImportError: If xbbg is not installed
        """
        try:
            import xbbg
            logger.info("Bloomberg connection ready (xbbg available)")
            self._connected = True
            return True
        except ImportError:
            logger.error("xbbg is not installed. Install with: pip install xbbg")
            raise ImportError("xbbg is not installed. Install with: pip install xbbg")
    
    def disconnect(self):
        """
        Disconnect from Bloomberg API.
        
        Note: With xbbg, there's no explicit disconnection needed.
        """
        self._connected = False
        logger.debug("Bloomberg connection closed (no-op for xbbg)")
    
    def is_connected(self) -> bool:
        """
        Check if Bloomberg connection is available.
        
        Returns:
            bool: True if xbbg is available, False otherwise
        """
        try:
            import xbbg
            return self._connected
        except ImportError:
            return False
    
    def get_bloomberg(self):
        """
        Get the Bloomberg API module (xbbg).
        
        Returns:
            module: The xbbg module
            
        Raises:
            ImportError: If xbbg is not installed
        """
        import xbbg
        return xbbg
