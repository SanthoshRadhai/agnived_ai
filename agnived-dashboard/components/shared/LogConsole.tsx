// components/shared/LogConsole.tsx
'use client';

import { useEffect, useRef } from 'react';
import { Activity, Trash2 } from 'lucide-react';

interface Log {
  time: string;
  message: string;
  level?: 'info' | 'success' | 'error' | 'warning';
}

interface LogConsoleProps {
  logs: Log[];
  onClear?: () => void;
  maxHeight?: string;
}

export default function LogConsole({ 
  logs, 
  onClear, 
  maxHeight = 'h-64' 
}: LogConsoleProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const getLogColor = (message: string) => {
    if (message.includes('✓') || message.includes('✅')) return 'text-green-400';
    if (message.includes('❌') || message.includes('ERROR')) return 'text-red-400';
    if (message.includes('⚠️') || message.includes('WARNING')) return 'text-orange-400';
    return 'text-slate-300';
  };

  return (
    <div className="panel-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold flex items-center gap-2">
          <Activity className="w-5 h-5 text-emerald-500" />
          Pipeline Logs
        </h3>
        <button
          onClick={onClear}
          className="p-1 rounded hover:bg-slate-700 transition-colors"
          aria-label="Clear logs"
        >
          <Trash2 className="w-4 h-4 text-slate-400" />
        </button>
      </div>

      <div
        className={`${maxHeight} overflow-y-auto font-mono text-xs space-y-1 bg-slate-950 rounded-lg p-4`}
      >
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-500">
            No logs yet. Run a pipeline to see output.
          </div>
        ) : (
          <>
            {logs.map((log, i) => (
              <div key={i} className="flex gap-2">
                <span className="text-slate-500">[{log.time}]</span>
                <span className={getLogColor(log.message)}>
                  {log.message}
                </span>
              </div>
            ))}
            <div ref={endRef} />
          </>
        )}
      </div>
    </div>
  );
}