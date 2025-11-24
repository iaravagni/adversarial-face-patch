import React, { useState, useEffect } from 'react';
import { Shield, ShieldAlert, User, AlertTriangle, Lock, Unlock, Server, FileCode, Cpu } from 'lucide-react';

const FaceRecognitionDemo = () => {
  const [imageType, setImageType] = useState('raw'); // 'raw' or 'patched'
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [defenseEnabled, setDefenseEnabled] = useState(false);
  const [scanLine, setScanLine] = useState(0);
  const [scanProgress, setScanProgress] = useState(0);
  const [gridOverlay, setGridOverlay] = useState(false);

  // Data from Backend
  const [attackers, setAttackers] = useState([]);
  const [employees, setEmployees] = useState([]);
  const [serverStatus, setServerStatus] = useState('checking'); // checking, online, offline

  const [selectedAttacker, setSelectedAttacker] = useState(null);
  const [selectedEmployee, setSelectedEmployee] = useState(null);

  const API_URL = "http://localhost:8000";

  // Initial Data Fetch
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`${API_URL}/info`);
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        setAttackers(data.attackers || []);
        setEmployees(data.employees || []);
        setServerStatus('online');
      } catch (error) {
        console.error("Failed to fetch info:", error);
        setServerStatus('offline');
        // Fallback data for UI testing if server is down
        setAttackers([
          { id: 999, name: 'Server Offline', db_id: 'err' }
        ]);
        setEmployees([
          { id: 999, name: 'Server Offline', db_id: 'err' }
        ]);
      }
    };

    fetchData();
  }, []);

  // Animation Effects
  useEffect(() => {
    if (isScanning) {
      const lineInterval = setInterval(() => {
        setScanLine(prev => (prev >= 100 ? 0 : prev + 2));
      }, 30);

      const progressInterval = setInterval(() => {
        setScanProgress(prev => Math.min(prev + 1, 99));
      }, 30); 

      const gridInterval = setInterval(() => {
        setGridOverlay(prev => !prev);
      }, 500);

      return () => {
        clearInterval(lineInterval);
        clearInterval(progressInterval);
        clearInterval(gridInterval);
      };
    } else {
      setScanLine(0);
      setScanProgress(0);
      setGridOverlay(false);
    }
  }, [isScanning]);

  // Reset to raw if target is deselected or if target is needed for patch
  useEffect(() => {
    // If no employee is selected OR if we are in patched mode without a target, revert to raw.
    if (!selectedEmployee && imageType === 'patched') {
      setImageType('raw');
    }
  }, [selectedEmployee, imageType]);

  const handleScan = async () => {
    if (!selectedAttacker) return;

    // If patched mode is selected, ensure a target is also selected.
    if (imageType === 'patched' && !selectedEmployee) {
        console.error("Cannot scan in patched mode without a target employee.");
        return;
    }


    setIsScanning(true);
    setScanResult(null);

    const targetId = selectedEmployee ? selectedEmployee.id : 0; 

    try {
      const response = await fetch(`${API_URL}/scan`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          attacker_id: selectedAttacker.id,
          target_id: targetId, // Use the resolved ID
          mode: imageType,
          defense: defenseEnabled
        }),
      });

      const result = await response.json();
      
      // Artificial delay for dramatic effect
      setTimeout(() => {
        setScanResult(result);
        setScanProgress(100);
        setIsScanning(false);
      }, 1500);

    } catch (error) {
      console.error("Scan failed:", error);
      setIsScanning(false);
      setScanResult({
        status: 'error',
        message: 'SYSTEM ERROR',
        detail: 'CONNECTION FAILED',
        color: 'red',
        confidence: 0
      });
    }
  };

  // Construct image URL for the scanner preview
  const getDisplayImageUrl = () => {
    if (!selectedAttacker) return null;
    
    const baseUrl = `${API_URL}/image`;
    const params = new URLSearchParams({
      attacker_id: selectedAttacker.id,
      mode: imageType,
      ts: Date.now()
    });
    
    // Target ID is only needed if the mode is 'patched'
    if (imageType === 'patched' && selectedEmployee) {
      params.append('target_id', selectedEmployee.id);
    } else if (imageType === 'patched' && !selectedEmployee) {
      return null;
    }

    return `${baseUrl}?${params.toString()}`;
  };


  return (
    <div className="h-screen w-screen bg-black text-white font-mono overflow-hidden flex flex-col p-2">
      
        {/* Top Bar */}
        <div className="bg-gradient-to-r from-slate-900 to-slate-800 border-2 border-cyan-500 rounded-lg p-2 mb-2 shadow-lg shadow-cyan-500/50 shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-cyan-500 rounded-full flex items-center justify-center animate-pulse">
                {defenseEnabled ? <Shield className="w-6 h-6 text-black" /> : <ShieldAlert className="w-6 h-6 text-black" />}
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-wider text-cyan-400">SECURE ID FACIAL RECOGNITION</h1>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-xs border border-slate-600 px-2 py-1 rounded bg-slate-800">
                <Server className={`w-3 h-3 ${serverStatus === 'online' ? 'text-green-400' : 'text-red-400'}`} />
                <span className={serverStatus === 'online' ? 'text-green-400' : 'text-red-400'}>
                  {serverStatus === 'online' ? 'API CONNECTED' : 'API OFFLINE'}
                </span>
              </div>

              <button
                onClick={() => {
                  setDefenseEnabled(!defenseEnabled);
                  setScanResult(null);
                }}
                className={`px-3 py-1 rounded-lg font-bold tracking-wider transition-all border-2 text-xs ${
                  defenseEnabled
                    ? 'bg-green-600 border-green-400 text-white shadow-lg shadow-green-500/50'
                    : 'bg-red-900 border-red-400 text-red-100 shadow-lg shadow-red-500/50'
                }`}
              >
                <div className="flex items-center gap-2">
                  {defenseEnabled ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                  <span>{defenseEnabled ? 'DEFENSE ACTIVE' : 'DEFENSE OFFLINE'}</span>
                </div>
              </button>
              <div className="text-right">
                <div className="text-cyan-400 text-[10px]">STATUS</div>
                <div className={`text-sm font-bold ${defenseEnabled ? 'text-green-400' : 'text-yellow-400'}`}>
                  {defenseEnabled ? 'PROTECTED' : 'VULNERABLE'}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-2 flex-1 min-h-0">
          {/* Left Panel - Controls */}
          <div className="xl:col-span-1 space-y-2 flex flex-col h-full overflow-hidden">
            
            {/* Subject Selection */}
            <div className="bg-slate-900 border-2 border-cyan-500 rounded-lg p-3 shadow-lg shadow-cyan-500/30 flex-[2] flex flex-col min-h-0">
              <div className="text-cyan-400 text-xs mb-2 tracking-wider shrink-0 flex items-center gap-2">
                <User className="w-3 h-3"/> SELECT ATTACKER (INPUT)
              </div>
              <div className="grid grid-cols-1 gap-2 mb-4 flex-1 min-h-0 overflow-y-auto">
                {attackers.map((attacker) => (
                  <button
                    key={attacker.id}
                    onClick={() => {
                      setSelectedAttacker(attacker);
                      setScanResult(null);
                    }}
                    className={`p-2 rounded border-2 transition-all text-center text-xs flex items-center justify-center gap-3 ${
                      selectedAttacker?.id === attacker.id
                        ? 'border-red-400 bg-red-900/30 shadow-lg shadow-red-500/30'
                        : 'border-slate-600 bg-slate-800 hover:border-red-600'
                    }`}
                  >
                    <span>{attacker.name}</span>
                  </button>
                ))}
              </div>

              <div className="text-cyan-400 text-xs mb-2 tracking-wider shrink-0 flex items-center gap-2">
                <User className="w-3 h-3"/> SELECT TARGET (IMPERSONATION GOAL)
              </div>
              <div className="grid grid-cols-1 gap-2 flex-[2] min-h-0 overflow-y-auto">
                {employees.map((employee) => (
                  <button
                    key={employee.id}
                    onClick={() => {
                      setSelectedEmployee(employee);
                      setScanResult(null);
                    }}
                    className={`p-2 rounded border-2 transition-all text-center flex items-center justify-center gap-3 ${
                      selectedEmployee?.id === employee.id
                        ? 'border-green-400 bg-green-900/30 shadow-lg shadow-green-500/30'
                        : 'border-slate-600 bg-slate-800 hover:border-green-600'
                    }`}
                  >
                    <span className="text-xs">{employee.name}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Image Mode & Patch Info */}
            <div className="bg-slate-900 border-2 border-cyan-500 rounded-lg p-3 shadow-lg shadow-cyan-500/30 flex-1 flex flex-col min-h-0">
              <div className="text-cyan-400 text-xs mb-2 tracking-wider text-center shrink-0">ATTACK CONFIGURATION</div>
              
              <div className="flex flex-col gap-2 flex-1">
                <button
                  onClick={() => {
                    setImageType('raw');
                    setScanResult(null);
                  }}
                  className={`rounded font-bold text-xs py-4 transition-all border-2 ${
                    imageType === 'raw'
                      ? 'bg-blue-600 border-blue-400 text-white'
                      : 'bg-slate-800 border-slate-600 text-slate-300 hover:border-blue-600'
                  }`}
                >
                  RAW INPUT
                </button>
                <button
                  onClick={() => {
                    setImageType('patched');
                    setScanResult(null);
                  }}
                  disabled={!selectedEmployee} 
                  className={`rounded font-bold text-xs py-4 transition-all border-2 ${
                    imageType === 'patched'
                      ? 'bg-red-600 border-red-400 text-white'
                      : !selectedEmployee 
                        ? 'bg-slate-800 border-slate-700 text-slate-600 cursor-not-allowed'
                        : 'bg-slate-800 border-slate-600 text-slate-300 hover:border-red-600'
                  }`}
                >
                  PATCHED INPUT
                </button>
              </div>
            </div>
            
          </div>

          {/* Center/Right - Scanner Display */}
          <div className="xl:col-span-2 h-full min-h-0">
            <div className="bg-slate-900 border-2 border-cyan-500 rounded-lg p-4 shadow-lg shadow-cyan-500/30 h-full flex flex-col">
              <div className="text-cyan-400 text-xs mb-2 tracking-wider flex justify-between shrink-0">
                <span>BIOMETRIC SCANNER</span>
                <span className="text-green-400 animate-pulse">● ONLINE</span>
              </div>

              {/* Scan Button */}
              <button
                disabled={!selectedAttacker || isScanning || serverStatus !== 'online' || (imageType === 'patched' && !selectedEmployee)}
                onClick={handleScan}
                className={`w-full py-3 rounded-lg font-bold text-md tracking-widest transition-all border-2 mb-3 shrink-0 ${
                  !selectedAttacker || isScanning || serverStatus !== 'online' || (imageType === 'patched' && !selectedEmployee)
                    ? 'bg-slate-800 border-slate-700 text-slate-600 cursor-not-allowed'
                    : 'bg-gradient-to-r from-cyan-600 to-blue-600 border-cyan-400 text-white hover:from-cyan-500 hover:to-blue-500 shadow-lg shadow-cyan-500/50 animate-pulse'
                }`}
              >
                {isScanning ? 'SCANNING...' : serverStatus === 'offline' ? 'SERVER OFFLINE' : 'INITIATE SCAN'}
              </button>

              {/* Scanner Area */}
              <div className="relative bg-black rounded-lg border-2 border-cyan-600 overflow-hidden flex-1 min-h-0 w-full">
                {selectedAttacker ? ( // Only check for attacker here, target is checked inside getDisplayImageUrl
                  <>
                    {/* Image */}
                    <img
                      src={getDisplayImageUrl()}
                      alt="Scan subject"
                      className="w-full h-full object-contain"
                      style={{ filter: isScanning ? 'brightness(1.2) contrast(1.1)' : 'none' }}
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />

                    {/* Scanning Overlay */}
                    {isScanning && (
                      <>
                        <div
                          className="absolute left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-lg shadow-cyan-500/50"
                          style={{
                            top: `${scanLine}%`,
                            transition: 'top 0.03s linear',
                            boxShadow: '0 0 20px 5px rgba(34, 211, 238, 0.8)'
                          }}
                        />

                        {gridOverlay && (
                          <div className="absolute inset-0 pointer-events-none">
                            <svg className="w-full h-full" style={{ opacity: 0.3 }}>
                              <defs>
                                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="cyan" strokeWidth="1"/>
                                </pattern>
                              </defs>
                              <rect width="100%" height="100%" fill="url(#grid)" />
                            </svg>
                          </div>
                        )}

                        <div className="absolute top-4 left-4 w-16 h-16 border-l-4 border-t-4 border-cyan-400 animate-pulse"></div>
                        <div className="absolute top-4 right-4 w-16 h-16 border-r-4 border-t-4 border-cyan-400 animate-pulse"></div>
                        <div className="absolute bottom-4 left-4 w-16 h-16 border-l-4 border-b-4 border-cyan-400 animate-pulse"></div>
                        <div className="absolute bottom-4 right-4 w-16 h-16 border-r-4 border-b-4 border-cyan-400 animate-pulse"></div>

                        <div className="absolute bottom-4 left-4 bg-black/80 px-3 py-2 rounded border border-cyan-400 text-xs">
                          <div className="text-cyan-400">ANALYZING: {imageType === 'patched' ? 'ADVERSARIAL PATTERN' : 'STANDARD BIOMETRICS'}</div>
                          <div className="text-cyan-400">MATCH SEARCH: {selectedEmployee ? selectedEmployee.name.toUpperCase() : "RAW MATCH"}</div>
                        </div>
                      </>
                    )}

                    {/* Results Overlay */}
                    {scanResult && !isScanning && (
                      <div className="absolute inset-0 bg-black/70 flex items-center justify-center backdrop-blur-sm p-4">
                        <div className={`text-center p-6 border-4 rounded-lg w-full max-w-lg ${
                          scanResult.color === 'green' 
                            ? 'border-green-400 bg-green-900/30' 
                            : 'border-red-400 bg-red-900/30'
                        }`}>
                          <div className={`text-4xl md:text-5xl font-bold mb-2 tracking-wider ${
                            scanResult.color === 'green' ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {scanResult.message}
                          </div>
                          <div className={`text-xl md:text-2xl font-bold mb-1 ${
                            scanResult.color === 'green' ? 'text-green-300' : 'text-red-300'
                          }`}>
                            {scanResult.detail}
                          </div>
                          {scanResult.subtext && (
                            <div className="text-sm md:text-lg text-slate-300 mt-1">{scanResult.subtext}</div>
                          )}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <div className="text-center text-slate-600">
                      <AlertTriangle className="w-12 h-12 mx-auto mb-2" />
                      <p className="text-lg">NO SUBJECT LOADED</p>
                      <p className="text-xs mt-1">Select attacker and target employee</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Status Bar */}
              <div className="mt-3 grid grid-cols-3 gap-3 text-[10px] shrink-0">
                <div className="bg-slate-800 p-2 rounded border border-cyan-600">
                  <div className="text-cyan-400 mb-1">MODE</div>
                  <div className={`font-bold ${imageType === 'patched' ? 'text-red-400' : 'text-blue-400'}`}>
                    {imageType === 'patched' ? 'PATCH INJECTION' : 'RAW IMAGE'}
                  </div>
                </div>
                <div className="bg-slate-800 p-2 rounded border border-cyan-600">
                  <div className="text-cyan-400 mb-1">DEFENSE</div>
                  <div className={`font-bold ${defenseEnabled ? 'text-green-400' : 'text-red-400'}`}>
                    {defenseEnabled ? 'ENABLED' : 'DISABLED'}
                  </div>
                </div>
                <div className="bg-slate-800 p-2 rounded border border-cyan-600">
                  <div className="text-cyan-400 mb-1">STATUS</div>
                  <div className="text-white font-bold">
                    {isScanning ? 'SCANNING' : scanResult ? 'COMPLETE' : 'READY'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
    </div>
  );
};

export default FaceRecognitionDemo;