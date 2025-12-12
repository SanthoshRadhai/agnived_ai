import React from "react";
import { Activity, Clock } from "lucide-react";

type Log = {
  time: string;
  message: string;
};

interface PipelineStatusProps {
  logs: Log[];
  setLogs: React.Dispatch<React.SetStateAction<Log[]>>;
}

export default function PipelineStatus({ logs, setLogs }: PipelineStatusProps) {
  return (
    <div className="rounded-2xl p-4 shadow-xl bg-green-950 border border-green-900">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold flex items-center gap-2 text-white">
          <Activity className="w-4 h-4 text-emerald-500" /> Pipeline Status
        </h3>
        <button onClick={() => setLogs([])} className="text-xs text-green-300">Clear</button>
      </div>
      <div className="h-40 overflow-y-auto p-2 font-mono text-xs space-y-1 bg-green-900">
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-green-400">
            <div className="text-center">
              <Clock className="w-6 h-6 mx-auto mb-2 opacity-50" />
              <p>Waiting for pipeline...</p>
            </div>
          </div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className="flex gap-2">
              <span className="text-green-400">[{log.time}]</span>
              <span>{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}