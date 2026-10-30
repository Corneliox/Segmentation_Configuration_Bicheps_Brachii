import React from 'react';
import { BiomechanicsSummary } from '../types/ultrasound';
import { Activity, Flame, ArrowUpRight, Clock } from 'lucide-react';

interface MetricsCardsProps {
  summary: BiomechanicsSummary;
}

export const MetricsCards: React.FC<MetricsCardsProps> = ({ summary }) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {/* Rest Thickness */}
      <div className="bg-[#0f172a] border border-[#334155] rounded-xl p-3 shadow-lg flex items-center justify-between">
        <div>
          <p className="text-xs text-[#94a3b8] font-medium uppercase tracking-wider">Rest Thickness</p>
          <p className="text-xl font-black text-[#38bdf8] mt-0.5">
            {summary.restThicknessMm.toFixed(2)} <span className="text-xs font-normal text-[#94a3b8]">mm</span>
          </p>
        </div>
        <div className="p-2.5 rounded-lg bg-[#0284c7]/10 text-[#38bdf8]">
          <Clock size={20} />
        </div>
      </div>

      {/* Peak Thickness */}
      <div className="bg-[#0f172a] border border-[#334155] rounded-xl p-3 shadow-lg flex items-center justify-between">
        <div>
          <p className="text-xs text-[#94a3b8] font-medium uppercase tracking-wider">Peak Thickness</p>
          <p className="text-xl font-black text-[#f43f5e] mt-0.5">
            {summary.peakThicknessMm.toFixed(2)} <span className="text-xs font-normal text-[#94a3b8]">mm</span>
          </p>
        </div>
        <div className="p-2.5 rounded-lg bg-[#f43f5e]/10 text-[#f43f5e]">
          <Flame size={20} />
        </div>
      </div>

      {/* Delta Contraction */}
      <div className="bg-[#0f172a] border border-[#334155] rounded-xl p-3 shadow-lg flex items-center justify-between">
        <div>
          <p className="text-xs text-[#94a3b8] font-medium uppercase tracking-wider">Delta Contraction</p>
          <p className="text-xl font-black text-[#4ade80] mt-0.5">
            +{summary.deltaThicknessMm.toFixed(2)} <span className="text-xs font-normal text-[#94a3b8]">mm</span>
          </p>
        </div>
        <div className="p-2.5 rounded-lg bg-[#4ade80]/10 text-[#4ade80]">
          <ArrowUpRight size={20} />
        </div>
      </div>

      {/* Strain / Increase % */}
      <div className="bg-[#0f172a] border border-[#334155] rounded-xl p-3 shadow-lg flex items-center justify-between">
        <div>
          <p className="text-xs text-[#94a3b8] font-medium uppercase tracking-wider">Contraction Strain</p>
          <p className="text-xl font-black text-[#f97316] mt-0.5">
            +{summary.percentageIncrease.toFixed(1)} <span className="text-xs font-normal text-[#94a3b8]">%</span>
          </p>
        </div>
        <div className="p-2.5 rounded-lg bg-[#f97316]/10 text-[#f97316]">
          <Activity size={20} />
        </div>
      </div>
    </div>
  );
};
