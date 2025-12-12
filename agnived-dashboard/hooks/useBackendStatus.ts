import { useEffect, useState } from 'react';
import { api } from '../lib/api/backend';

export function useBackendStatus() {
  const [isConnected, setIsConnected] = useState(false);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      setChecking(true);
      const healthy = await api.healthCheck();
      setIsConnected(healthy);
      setChecking(false);
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s

    return () => clearInterval(interval);
  }, []);

  return { isConnected, checking };
}