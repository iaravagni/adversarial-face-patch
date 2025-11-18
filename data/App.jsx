import React, { useState, useEffect } from 'react';
import { Shield, ShieldAlert, Upload, User, AlertTriangle, Lock, Unlock } from 'lucide-react';
import './App.css';  

const FaceRecognitionDemo = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedPatch, setSelectedPatch] = useState(null);
  const [imageType, setImageType] = useState('raw');
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [defenseEnabled, setDefenseEnabled] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [scanLine, setScanLine] = useState(0);
  const [scanProgress, setScanProgress] = useState(0);
  const [gridOverlay, setGridOverlay] = useState(false);
  const [availablePatches, setAvailablePatches] = useState([]);
  const [apiStatus, setApiStatus] = useState('connecting');

  // API endpoint - adjust based on your setup
  const API_URL = 'http://localhost:8000';

  // Pre-loaded test subjects (these would come from your dataset)
  const preloadedImages = [
    { id: 1, name: 'Attacker Profile', raw: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="400"%3E%3Crect fill="%23444" width="400" height="400"/%3E%3Ccircle cx="200" cy="150" r="60" fill="%23666"/%3E%3Cellipse cx="200" cy="280" rx="80" ry="100" fill="%23666"/%3E%3Ccircle cx="180" cy="140" r="8" fill="%23222"/%3E%3Ccircle cx="220" cy="140" r="8" fill="%23222"/%3E%3Cpath d="M 180 180 Q 200 190 220 180" stroke="%23222" fill="none" stroke-width="3"/%3E%3C/svg%3E' },
    { id: 2, name: 'Test Subject A', raw: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="400"%3E%3Crect fill="%23555" width="400" height="400"/%3E%3Ccircle cx="200" cy="150" r="65" fill="%23777"/%3E%3Cellipse cx="200" cy="280" rx="85" ry="105" fill="%23777"/%3E%3Ccircle cx="180" cy="145" r="9" fill="%23111"/%3E%3Ccircle cx="220" cy="145" r="9" fill="%23111"/%3E%3Cpath d="M 175 185 Q 200 195 225 185" stroke="%23111" fill="none" stroke-width="4"/%3E%3C/svg%3E' },
    { id: 3, name: 'Test Subject B', raw: 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="400"%3E%3Crect fill="%23333" width="400" height="400"/%3E%3Ccircle cx="200" cy="155" r="55" fill="%23555"/%3E%3Cellipse cx="200" cy="285" rx="75" ry="95" fill="%23555"/%3E%3Ccircle cx="185" cy="145" r="7" fill="%23000"/%3E%3Ccircle cx="215" cy="145" r="7" fill="%23000"/%3E%3Cpath d="M 185 180 Q 200 188 215 180" stroke="%23000" fill="none" stroke-width="3"/%3E%3C/svg%3E' }
  ];

  // Check API status on mount
  useEffect(() => {
    checkApiStatus();
    loadAvailablePatches();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/`);
      if (response.ok) {
        setApiStatus('connected');
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      setApiStatus('offline');
      console.error('API connection error:', error);
    }
  };

  const loadAvailablePatches = async () => {
    try {
      const response = await fetch(`${API_URL}/api/patches`);
      if (response.ok) {
        const patches = await response.json();
        setAvailablePatches(patches);
        if (patches.length > 0) {
          setSelectedPatch(patches[0]);
        }
      }
    } catch (error) {
      console.error('Error loading patches:', error);
      // Use mock data if API unavailable
      setAvailablePatches([
        { name: 'patch_employee_1_r25', type: 'circular', size: 50, target_employee: 'Employee_1', success_rate: 85.0 },
        { name: 'patch_employee_2_r25', type: 'circular', size: 50, target_employee: 'Employee_2', success_rate: 78.5 },
        { name: 'patch_employee_3_r25', type: 'circular', size: 50, target_employee: 'Employee_3', success_rate: 92.3 }
      ]);
      setSelectedPatch({ name: 'patch_employee_1_r25', type: 'circular', size: 50, target_employee: 'Employee_1', success_rate: 85.0 });
    }
  };

  useEffect(() => {
    if (isScanning) {
      const lineInterval = setInterval(() => {
        setScanLine(prev => (prev >= 100 ? 0 : prev + 2));
      }, 30);

      const progressInterval = setInterval(() => {
        setScanProgress(prev => Math.min(prev + 1, 100));
      }, 20);

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

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        const imageData = event.target.result;
        setUploadedImage(imageData);
        setSelectedImage({ id: 'uploaded', name: file.name, raw: imageData });
        setScanResult(null);

        // Optionally upload to backend
        if (apiStatus === 'connected') {
          try {
            const formData = new FormData();
            formData.append('file', file);
            await fetch(`${API_URL}/api/upload-image`, {
              method: 'POST',
              body: formData
            });
          } catch (error) {
            console.error('Upload error:', error);
          }
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleScan = async () => {
    setIsScanning(true);
    setScanResult(null);

    // If API is connected, use backend
    if (apiStatus === 'connected' && selectedImage) {
      try {
        const response = await fetch(`${API_URL}/api/scan`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image_data: selectedImage.raw,
            image_type: imageType,
            defense_enabled: defenseEnabled,
            patch_name: selectedPatch?.name
          })
        });

        const result = await response.json();
        
        setTimeout(() => {
          setScanResult(result);
          setIsScanning(false);
        }, 2500);
        
      } catch (error) {
        console.error('Scan error:', error);
        // Fallback to demo mode
        performDemoScan();
      }
    } else {
      // Demo mode (no API)
      performDemoScan();
    }
  };

  const performDemoScan = () => {
    setTimeout(() => {
      if (imageType === 'raw') {
        setScanResult({
          status: 'unknown',
          message: 'ACCESS DENIED',
          detail: 'UNRECOGNIZED SUBJECT',
          confidence: 0,
          color: 'red'
        });
      } else {
        if (defenseEnabled) {
          setScanResult({
            status: 'threat',
            message: 'THREAT DETECTED',
            detail: 'ADVERSARIAL PATTERN IDENTIFIED',
            subtext: 'Patch-based attack blocked',
            confidence: 0,
            color: 'red'
          });
        } else {
          const targetEmployee = selectedPatch?.target_employee || 'Employee #7';
          setScanResult({
            status: 'recognized',
            message: 'ACCESS GRANTED',
            detail: targetEmployee.toUpperCase(),
            subtext: 'SARAH CHEN - ENGINEERING DEPT.',
            confidence: 94,
            color: 'green'
          });
        }
      }
      
      setIsScanning(false);
    }, 2500);
  };

  const getCurrentImage = () => {
    if (!selectedImage) return null;
    // In production, this would apply the patch via backend
    return selectedImage.raw;
  };

  return (
    <div className="min-h-screen bg-black text-white p-4 font-mono">
      <div className="max-w-7xl mx-auto">
        {/* Top Bar */}
        <div className="bg-gradient-to-r from-slate-900 to-slate-800 border-2 border-cyan-500 rounded-lg p-4 mb-4 shadow-lg shadow-cyan-500/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-cyan-500 rounded-full flex items-center justify-center animate-pulse">
                {defenseEnabled ? <Shield className="w-7 h-7 text-black" /> : <ShieldAlert className="w-7 h-7 text-black" />}
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-wider text-cyan-400">SECUREID FACIAL RECOGNITION</h1>
                <p className="text-xs text-cyan-300 tracking-widest">SECURITY CLEARANCE SYSTEM v3.7.2</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-cyan-400 text-sm">API STATUS</div>
              <div className={`text-xl font-bold ${
                apiStatus === 'connected' ? 'text-green-400' : 
                apiStatus === 'connecting' ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {apiStatus.toUpperCase()}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1 space-y-4">
            {/* Defense Control */}
            <div className="bg-slate-900 border-2 border-cyan-500 rounded-lg p-4 shadow-lg shadow-cyan-500/30">
              <div className="text-cyan-400 text-sm mb-3 tracking-wider">DEFENSE SYSTEM</div>
              <button
                onClick={() => {
                  setDefenseEnabled(!defenseEnabled);
                  setScanResult(null);
                }}
                className={`w-full py-4 rounded-lg font-bold tracking-wider transition-all border-2 ${
                  defenseEnabled
                    ? 'bg-green-600 border-green-400 text-white shadow-lg shadow-green-500/50'
                    : 'bg-red-900 border-red-400 text-red-100 shadow-lg shadow-red-500/50'
                }`}
              >
                <div className="flex items-center justify-center gap-2">
                  {defenseEnabled ? <Lock className="w-5 h-5" /> : <Unlock className="w-5 h-5" />}
                  {defenseEnabled ? 'DEFENSE ACTIVE' : 'DEFENSE OFFLINE'}
                </div>
              </button>
              <div className="mt-3 text-xs text-center text-slate-400">
                {defenseEnabled ? 'Anti-adversarial protection enabled' : 'System vulnerable to attacks'}
              </div>
            </div>

            {/* Patch Selection */}
            <div className="bg-slate-900 border-2 border-cyan-500 rounded-lg p-4 shadow-lg shadow-cyan-500/30">
              <div className="text-cyan-400 text-sm mb-3 tracking-wider">ADVERSARIAL PATCH</div>
              
              <div className="grid grid-cols-1 gap-2 max-h-48 overflow-y-auto">
                {availablePatches.map((patch) => (
                  <button
                    key={patch.name}
                    onClick={() => {
                      setSelectedPatch(patch);
                      setScanResult(null);
                    }}
                    className={`p-3 rounded border-2 transition-all text-left text-xs ${
                      selectedPatch?.name === patch.name
                        ? 'border-red-400 bg-red-900/30 shadow-lg shadow-red-500/30'
                        : 'border-slate-600 bg-slate-800 hover:border-red-600'
                    }`}
                  >
                    <div className="font-bold text-red-400">{patch.name}</div>
                    <div className="text-slate-400 mt-1">
                      Target: {patch.target_employee} | 
                      Success: {patch.success_rate.toFixed(1)}%
                    </div>
                  </button>
                ))}
              </div>
              
              {availablePatches.length === 0 && (
                <div className="text-center text-slate-500 py-4 text-xs">
                  No patches loaded. Run optimization scripts first.
                </div>
              )}
            </div>

            {/* Image Selection */}
            <div className="bg-slate-900 border-2 border-cyan-500 rounded-lg p-4 shadow-lg shadow-cyan-500/30">
              <div className="text-cyan-400 text-sm mb-3 tracking-wider">SUBJECT SELECTION</div>
              
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="block w-full py-3 px-4 bg-slate-800 border border-cyan-500 rounded text-center cursor-pointer hover:bg-slate-700 transition-all mb-3 text-sm"
              >
                <Upload className="w-4 h-4 inline mr-2" />
                UPLOAD IMAGE
              </label>

              <div className="grid grid-cols-1 gap-2">
                {preloadedImages.map((img) => (
                  <button
                    key={img.id}
                    onClick={() => {
                      setSelectedImage(img);
                      setScanResult(null);
                    }}
                    className={`p-3 rounded border-2 transition-all text-left text-sm ${
                      selectedImage?.id === img.id
                        ? 'border-cyan-400 bg-cyan-900/30 shadow-lg shadow-cyan-500/30'
                        : 'border-slate-600 bg-slate-800 hover:border-cyan-600'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <User className="w-5 h-5 text-cyan-400" />
                      <span>{img.name}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Image Type */}
            <div className="bg-slate-900 border-2 border-cyan-500 rounded-lg p-4 shadow-lg shadow-cyan-500/30">
              <div className="text-cyan-400 text-sm mb-3 tracking-wider">IMAGE MODE</div>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => {
                    setImageType('raw');
                    setScanResult(null);
                  }}
                  className={`py-3 rounded font-bold text-sm transition-all border-2 ${
                    imageType === 'raw'
                      ? 'bg-blue-600 border-blue-400 text-white'
                      : 'bg-slate-800 border-slate-600 text-slate-300 hover:border-blue-600'
                  }`}
                >
                  RAW
                </button>
                <button
                  onClick={() => {
                    setImageType('patched');
                    setScanResult(null);
                  }}
                  className={`py-3 rounded font-bold text-sm transition-all border-2 ${
                    imageType === 'patched'
                      ? 'bg-red-600 border-red-400 text-white'
                      : 'bg-slate-800 border-slate-600 text-slate-300 hover:border-red-600'
                  }`}
                >
                  PATCHED
                </button>
              </div>
            </div>

            {/* Scan Button */}
            <button
              onClick={handleScan}
              disabled={!selectedImage || isScanning}
              className={`w-full py-6 rounded-lg font-bold text-lg tracking-widest transition-all border-2 ${
                !selectedImage || isScanning
                  ? 'bg-slate-800 border-slate-700 text-slate-600 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-600 to-blue-600 border-cyan-400 text-white hover:from-cyan-500 hover:to-blue-500 shadow-lg shadow-cyan-500/50 animate-pulse'
              }`}
            >
              {isScanning ? 'SCANNING...' : 'INITIATE SCAN'}
            </button>
          </div>

          {/* Center/Right - Scanner Display */}
          <div className="lg:col-span-2">
            <div className="bg-slate-900 border-2 border-cyan-500 rounded-lg p-6 shadow-lg shadow-cyan-500/30 h-full">
              <div className="text-cyan-400 text-sm mb-4 tracking-wider flex justify-between">
                <span>BIOMETRIC SCANNER</span>
                <span className="text-green-400 animate-pulse">● ONLINE</span>
              </div>

              {/* Scanner Area */}
              <div className="relative bg-black rounded-lg border-2 border-cyan-600 overflow-hidden" style={{ aspectRatio: '4/3' }}>
                {selectedImage ? (
                  <>
                    {/* Image */}
                    <img
                      src={getCurrentImage()}
                      alt="Scan subject"
                      className="w-full h-full object-contain"
                      style={{ filter: isScanning ? 'brightness(1.2) contrast(1.1)' : 'none' }}
                    />

                    {/* Scanning Overlay */}
                    {isScanning && (
                      <>
                        {/* Scanning Line */}
                        <div
                          className="absolute left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-lg shadow-cyan-500/50"
                          style={{
                            top: `${scanLine}%`,
                            transition: 'top 0.03s linear',
                            boxShadow: '0 0 20px 5px rgba(34, 211, 238, 0.8)'
                          }}
                        />

                        {/* Grid Overlay */}
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

                        {/* Corner Brackets */}
                        <div className="absolute top-4 left-4 w-16 h-16 border-l-4 border-t-4 border-cyan-400 animate-pulse"></div>
                        <div className="absolute top-4 right-4 w-16 h-16 border-r-4 border-t-4 border-cyan-400 animate-pulse"></div>
                        <div className="absolute bottom-4 left-4 w-16 h-16 border-l-4 border-b-4 border-cyan-400 animate-pulse"></div>
                        <div className="absolute bottom-4 right-4 w-16 h-16 border-r-4 border-b-4 border-cyan-400 animate-pulse"></div>

                        {/* Scanning Info */}
                        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-black/80 px-4 py-2 rounded border border-cyan-400">
                          <div className="text-cyan-400 text-sm font-bold">ANALYZING BIOMETRICS</div>
                          <div className="text-xs text-cyan-300">Progress: {scanProgress}%</div>
                        </div>

                        {/* Random scanning data */}
                        <div className="absolute bottom-4 left-4 bg-black/80 px-3 py-2 rounded border border-cyan-400 text-xs">
                          <div className="text-cyan-400">FACIAL POINTS: 128</div>
                          <div className="text-cyan-400">IRIS SCAN: ACTIVE</div>
                          <div className="text-cyan-400">MATCH SEARCH: RUNNING</div>
                        </div>
                      </>
                    )}

                    {/* Results Overlay */}
                    {scanResult && !isScanning && (
                      <div className="absolute inset-0 bg-black/70 flex items-center justify-center backdrop-blur-sm">
                        <div className={`text-center p-8 border-4 rounded-lg ${
                          scanResult.color === 'green' 
                            ? 'border-green-400 bg-green-900/30' 
                            : 'border-red-400 bg-red-900/30'
                        }`}>
                          <div className={`text-6xl font-bold mb-4 tracking-wider ${
                            scanResult.color === 'green' ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {scanResult.message}
                          </div>
                          <div className={`text-3xl font-bold mb-2 ${
                            scanResult.color === 'green' ? 'text-green-300' : 'text-red-300'
                          }`}>
                            {scanResult.detail}
                          </div>
                          {scanResult.subtext && (
                            <div className="text-xl text-slate-300 mt-2">{scanResult.subtext}</div>
                          )}
                          {scanResult.confidence > 0 && (
                            <div className="mt-6">
                              <div className="text-green-400 text-2xl font-bold">
                                MATCH: {scanResult.confidence}%
                              </div>
                              <div className="w-64 mx-auto mt-3 h-4 bg-slate-700 rounded-full overflow-hidden border-2 border-green-400">
                                <div
                                  className="h-full bg-gradient-to-r from-green-600 to-green-400 transition-all duration-1000"
                                  style={{ width: `${scanResult.confidence}%` }}
                                ></div>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <div className="text-center text-slate-600">
                      <AlertTriangle className="w-16 h-16 mx-auto mb-4" />
                      <p className="text-xl">NO SUBJECT LOADED</p>
                      <p className="text-sm mt-2">Select or upload an image to begin</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Status Bar */}
              <div className="mt-4 grid grid-cols-4 gap-4 text-xs">
                <div className="bg-slate-800 p-3 rounded border border-cyan-600">
                  <div className="text-cyan-400 mb-1">MODE</div>
                  <div className="text-white font-bold">{imageType.toUpperCase()}</div>
                </div>
                <div className="bg-slate-800 p-3 rounded border border-cyan-600">
                  <div className="text-cyan-400 mb-1">PATCH</div>
                  <div className="text-white font-bold truncate">{selectedPatch?.name.split('_')[2].toUpperCase() || 'NONE'}</div>
                </div>
                <div className="bg-slate-800 p-3 rounded border border-cyan-600">
                  <div className="text-cyan-400 mb-1">DEFENSE</div>
                  <div className={`font-bold ${defenseEnabled ? 'text-green-400' : 'text-red-400'}`}>
                    {defenseEnabled ? 'ENABLED' : 'DISABLED'}
                  </div>
                </div>
                <div className="bg-slate-800 p-3 rounded border border-cyan-600">
                  <div className="text-cyan-400 mb-1">STATUS</div>
                  <div className="text-white font-bold">
                    {isScanning ? 'SCANNING' : scanResult ? 'COMPLETE' : 'READY'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Info */}
        <div className="mt-4 text-center text-xs text-slate-600">
          <p>SECUREID CYBERSECURITY DEMONSTRATION SYSTEM</p>
          <p className="mt-1">Adversarial Patch Attack & Defense Testing Platform</p>
        </div>
      </div>
    </div>
  );
};

export default FaceRecognitionDemo;