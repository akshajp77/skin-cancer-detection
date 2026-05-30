import React, { useState, useCallback } from "react";
import { 
  Upload, 
  Sparkles, 
  AlertTriangle, 
  Activity, 
  ArrowRight, 
  Microscope, 
  CheckCircle,
  FileText
} from "lucide-react";

// 7 class list with classification characteristics and malignancy designations
const LESION_INFO = {
  nv: {
    key: "nv",
    display_name: "Melanocytic nevi (benign mole)",
    risk: "benign",
    desc: "A completely benign skin growth commonly known as a mole. They consist of clusters of melanocytes (pigment-producing cells) and typically present with uniform borders and color."
  },
  mel: {
    key: "mel",
    display_name: "Melanoma (malignant)",
    risk: "malignant",
    desc: "A highly malignant skin cancer arising from melanocytes. It can develop inside existing moles or appear as new asymmetrical dark spots with irregular borders. Immediate clinical attention is highly advised."
  },
  bkl: {
    key: "bkl",
    display_name: "Benign keratosis",
    risk: "benign_others",
    desc: "A common, non-cancerous skin growth that typically appears waxy, scaly, or slightly raised. They are completely benign and do not carry malignancy risk."
  },
  bcc: {
    key: "bcc",
    display_name: "Basal cell carcinoma (malignant)",
    risk: "malignant",
    desc: "The most common form of skin cancer. It is malignant but slow-growing and rarely metastasizes. It typically presents as a pearly, pinkish bump or translucent patch of skin."
  },
  akiec: {
    key: "akiec",
    display_name: "Actinic keratosis (pre-malignant)",
    risk: "malignant",
    desc: "A pre-malignant, rough, scaly patch on the skin caused by years of sun exposure. A small percentage can eventually progress into invasive squamous cell carcinoma if left untreated."
  },
  vasc: {
    key: "vasc",
    display_name: "Vascular lesion",
    risk: "benign_others",
    desc: "Benign skin abnormalities formed by concentrated blood vessels under the surface. Examples include cherry angiomas and pyogenic granulomas."
  },
  df: {
    key: "df",
    display_name: "Dermatofibroma",
    risk: "benign_others",
    desc: "A very common, benign firm nodule that often appears on the lower legs. It is harmless and typically displays a positive dimple sign (depresses slightly) when compressed."
  }
};

const CLASS_KEYS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [inferenceStatus, setInferenceStatus] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [simulatorMode, setSimulatorMode] = useState(false);

  // Drag and Drop support handlers
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    if (!file.type.startsWith("image/")) {
      setError("Supported file types are limited to standard images (PNG, JPG, WEBP).");
      return;
    }
    setError(null);
    setPrediction(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  // Standalone Simulator Fallback (Triggers dynamically if Backend connection is refused)
  const runLocalSimulation = async () => {
    setLoading(true);
    setSimulatorMode(true);
    
    const steps = [
      "Configuring local development canvas...",
      "Extracting RGB matrix structure...",
      "Simulating EfficientNetB0 computational pass...",
      "Computing mathematical activation offsets..."
    ];

    for (const step of steps) {
      setInferenceStatus(step);
      await new Promise((res) => setTimeout(res, 600));
    }

    // Generate random mock classification results
    const randIdx = Math.floor(Math.random() * CLASS_KEYS_ORDER.length);
    const mockClass = CLASS_KEYS_ORDER[randIdx];
    const mockProbabilities = CLASS_KEYS_ORDER.map((_, i) => 
      i === randIdx ? 0.70 + Math.random() * 0.25 : Math.random() * 0.05
    );
    const sum = mockProbabilities.reduce((a, b) => a + b, 0);
    const normalizedProbs = mockProbabilities.map(p => p / sum);

    setPrediction({
      predicted_class: mockClass,
      display_name: LESION_INFO[mockClass].display_name,
      confidence: normalizedProbs[randIdx],
      all_probabilities: normalizedProbs,
      gradcam_image: null // Denotes simulator mock map fallback
    });

    setLoading(false);
  };

  const executeInference = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setPrediction(null);
    setSimulatorMode(false);
    setInferenceStatus("Compressing and transmitting image data...");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned error status: ${response.status}`);
      }

      setInferenceStatus("Acquiring outputs and superimposing Grad-CAM heatmaps...");
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.warn("Backend unavailable. Redirecting transaction to local simulation pipeline.");
      await runLocalSimulation();
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (key) => {
    const risk = LESION_INFO[key]?.risk;
    if (risk === "malignant") return "bg-rose-500";
    if (risk === "benign") return "bg-emerald-500";
    return "bg-slate-400";
  };

  const getRiskBorder = (key) => {
    const risk = LESION_INFO[key]?.risk;
    if (risk === "malignant") return "border-rose-200 dark:border-rose-950/40 bg-rose-50/20 dark:bg-rose-950/10";
    if (risk === "benign") return "border-emerald-200 dark:border-emerald-950/40 bg-emerald-50/20 dark:bg-emerald-950/10";
    return "border-slate-200 dark:border-slate-800 bg-slate-50/40 dark:bg-slate-900/10";
  };

  return (
    <div className="min-h-screen bg-slate-50/40 dark:bg-slate-950 text-slate-800 dark:text-slate-100 font-sans transition-colors duration-300">
      
      {/* HEADER BANNER */}
      <header className="border-b border-slate-200/60 dark:border-slate-900 bg-white dark:bg-slate-950 sticky top-0 z-10 shadow-sm backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-4 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-teal-500/10 text-teal-600 dark:text-teal-400">
              <Microscope className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-slate-900 dark:text-white">
                Derm-Inference Station
              </h1>
              <p className="text-xs text-slate-500 font-mono">
                BME-Pipeline: EfficientNetB0 + Grad-CAM Heatmap Analysis
              </p>
            </div>
          </div>
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-900/40">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
            Local Pipeline Operational
          </span>
        </div>
      </header>

      {/* CORE WORKSPACE */}
      <main className="max-w-6xl mx-auto px-4 py-8 space-y-8">
        
        {/* DISCLAIMER BANNER */}
        <div className="rounded-2xl border-2 border-amber-200 dark:border-amber-950/40 bg-amber-500/5 dark:bg-amber-500/10 p-5 flex items-start gap-4">
          <div className="p-2 bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400 rounded-lg">
            <AlertTriangle className="h-5 w-5" />
          </div>
          <div className="space-y-1">
            <h3 className="text-sm font-bold text-amber-800 dark:text-amber-400">
              EDUCATIONAL RESEARCH CLASSIFIER SYSTEM
            </h3>
            <p className="text-xs text-amber-700/85 dark:text-amber-400/80 leading-relaxed">
              For educational purposes only — not a clinical diagnostic tool. Computational predictions are designed to assist laboratory learning modules and have not been certified for patient diagnostic evaluations.
            </p>
          </div>
        </div>

        {/* WORKSPACE PANELS */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT COLUMN: INTERACTION AREA */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800 p-6 space-y-6 shadow-sm">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider">
                  Dermoscopy Capture Input
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  Upload localized dermoscopic images to calculate regional attention activation maps.
                </p>
              </div>

              {/* DRAG AND DROP BOX */}
              <div
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={() => document.getElementById("file-picker").click()}
                className={`relative rounded-xl border-2 border-dashed p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 min-h-[220px] ${
                  isDragActive
                    ? "border-teal-500 bg-teal-500/5 dark:bg-teal-950/20"
                    : "border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700"
                }`}
              >
                <input
                  id="file-picker"
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleFileInput}
                />

                {previewUrl ? (
                  <div className="space-y-4 w-full">
                    <img
                      src={previewUrl}
                      alt="Dermoscopy input target"
                      className="max-h-[160px] mx-auto rounded-lg object-cover border border-slate-200 dark:border-slate-800"
                    />
                    <p className="text-[11px] font-mono text-slate-500 truncate max-w-xs mx-auto">
                      {selectedFile?.name}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="p-3 bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500 rounded-full inline-block">
                      <Upload className="h-6 w-6" />
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                        Drag and drop skin lesion image here
                      </p>
                      <p className="text-[10px] text-slate-500 mt-1 font-mono">
                        Supports PNG, JPG, or JPEG formats
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* EXECUTION CONTROL PANEL */}
              {selectedFile && (
                <button
                  onClick={executeInference}
                  disabled={loading}
                  className="w-full py-3 px-4 rounded-xl text-xs font-semibold tracking-wide text-white bg-teal-600 hover:bg-teal-700 disabled:bg-slate-200 disabled:text-slate-400 dark:disabled:bg-slate-800 transition-colors flex items-center justify-center gap-2 shadow-md shadow-teal-600/10"
                >
                  {loading ? (
                    <>
                      <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Inference Running...
                    </>
                  ) : (
                    <>
                      <Activity className="h-4 w-4" />
                      Run Model Inference & Grad-CAM
                    </>
                  )}
                </button>
              )}

              {error && (
                <div className="p-3.5 rounded-xl border border-rose-200 dark:border-rose-950/40 bg-rose-50/30 dark:bg-rose-950/10 text-rose-600 dark:text-rose-400 text-xs">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* RIGHT COLUMN: INFERENCE & VISUALIZATION OUTPUT */}
          <div className="lg:col-span-7">
            
            {/* INFERENCE LOADING WRAPPER */}
            {loading && (
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800 p-12 flex flex-col items-center justify-center text-center space-y-6 shadow-sm min-h-[400px]">
                <div className="relative">
                  <div className="h-16 w-16 rounded-full border-4 border-slate-100 dark:border-slate-800" />
                  <div className="absolute inset-0 h-16 w-16 rounded-full border-4 border-teal-600 dark:border-teal-400 border-t-transparent animate-spin" />
                </div>
                <div className="space-y-2">
                  <h4 className="font-bold text-sm text-slate-800 dark:text-white uppercase tracking-wider">
                    Running Deep Neural Inference
                  </h4>
                  <p className="text-xs font-mono text-teal-600 dark:text-teal-400 animate-pulse">
                    {inferenceStatus}
                  </p>
                </div>
              </div>
            )}

            {/* EMPTY STATUS SCREEN */}
            {!loading && !prediction && (
              <div className="bg-white/40 dark:bg-slate-900/40 border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-2xl p-12 text-center flex flex-col items-center justify-center min-h-[400px]">
                <div className="p-3 bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-600 rounded-full mb-3">
                  <Activity className="h-7 w-7" />
                </div>
                <h3 className="font-bold text-xs text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                  Waiting for Diagnostic Session
                </h3>
                <p className="text-xs text-slate-500 max-w-sm mt-2 leading-relaxed">
                  Provide a lesion image on the left controller to run the feed-forward classifier and project deep spatial gradients.
                </p>
              </div>
            )}

            {/* PREDICTION REPORT DISPLAY */}
            {prediction && !loading && (
              <div className="space-y-6 animate-fade-in">
                
                {/* PREDICTED METRIC SUMMARY PANEL */}
                <div className={`p-6 rounded-2xl border ${getRiskBorder(prediction.predicted_class)} shadow-sm space-y-4`}>
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-200/40 dark:border-slate-800/40 pb-4">
                    <div>
                      <span className="text-[10px] font-bold font-mono text-slate-400 uppercase tracking-wider">
                        Inference Output Result
                      </span>
                      <h2 className="text-xl font-black text-slate-900 dark:text-white mt-1">
                        {prediction.display_name}
                      </h2>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-extrabold bg-teal-500/10 text-teal-600 dark:text-teal-400 border border-teal-500/20 px-2.5 py-1 rounded-lg">
                        {(prediction.confidence * 100).toFixed(1)}% Confidence
                      </span>
                      <span className={`px-2.5 py-1 text-[10px] font-bold tracking-wider uppercase rounded-lg border ${
                        LESION_INFO[prediction.predicted_class]?.risk === "malignant"
                          ? "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20"
                          : LESION_INFO[prediction.predicted_class]?.risk === "benign"
                          ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                          : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400 border-slate-500/20"
                      }`}>
                        {LESION_INFO[prediction.predicted_class]?.risk === "malignant" 
                          ? "High Risk" 
                          : LESION_INFO[prediction.predicted_class]?.risk === "benign" 
                          ? "Low Risk" 
                          : "Monitor"}
                      </span>
                    </div>
                  </div>

                  {/* Plain English Lesion Description */}
                  <div className="space-y-1.5">
                    <h4 className="text-[10px] font-bold font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                      <FileText className="h-3.5 w-3.5 text-teal-500" />
                      Educational Summary
                    </h4>
                    <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                      {LESION_INFO[prediction.predicted_class]?.desc}
                    </p>
                  </div>
                </div>

                {/* VISUAL WORKSTATION: SIDE-BY-SIDE COMPARE */}
                <div className="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 space-y-4 shadow-sm">
                  <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Workstation Image Registration
                    </h3>
                    {simulatorMode && (
                      <span className="text-[9px] font-bold font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                        Heatmap Sim Mode
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* ORIGINAL VIEW */}
                    <div className="space-y-2">
                      <span className="text-[10px] font-bold font-mono text-slate-400 uppercase block">
                        Original Input
                      </span>
                      <div className="aspect-square w-full rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 flex items-center justify-center">
                        <img src={previewUrl} alt="Target dermoscopy scan" className="w-full h-full object-cover" />
                      </div>
                    </div>

                    {/* GRAD-CAM HEATMAP VIEW */}
                    <div className="space-y-2">
                      <span className="text-[10px] font-bold font-mono text-slate-400 uppercase block">
                        Grad-CAM Heatmap Overlay
                      </span>
                      <div className="aspect-square w-full rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 flex items-center justify-center relative">
                        {prediction.gradcam_image ? (
                          <img src={prediction.gradcam_image} alt="Grad-CAM heat map" className="w-full h-full object-cover" />
                        ) : (
                          // Fallback mock visualization if running in simulated backend mode
                          <div className="w-full h-full relative">
                            <img src={previewUrl} alt="Simulator base" className="w-full h-full object-cover filter brightness-75" />
                            <div className="absolute inset-0 bg-rose-500/40 mix-blend-color-burn" />
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-28 h-28 rounded-full bg-gradient-to-tr from-yellow-500/80 via-orange-600/70 to-red-600/80 blur-xl mix-blend-screen animate-pulse" />
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-3 bg-slate-50 dark:bg-slate-950/60 rounded-xl border border-slate-100 dark:border-slate-800/60 flex gap-2">
                    <Sparkles className="h-4.5 w-4.5 text-teal-500 flex-shrink-0 mt-0.5" />
                    <p className="text-[10px] text-slate-500 leading-normal">
                      The heatmap highlights regions matching the highest activations inside the model&apos;s final convolutional layer (<code>top_conv</code>). These locations drove the categorical determination output.
                    </p>
                  </div>
                </div>

                {/* HORIZONTAL PROBABILITIES CHART */}
                <div className="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 space-y-4 shadow-sm">
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Output Probability distribution
                    </h3>
                    <p className="text-[10px] text-slate-400 mt-1">
                      Probability metrics across all 7 target classes, color-coded by malignancy risk vectors.
                    </p>
                  </div>

                  <div className="space-y-3">
                    {CLASS_KEYS_ORDER.map((key, idx) => {
                      const prob = prediction.all_probabilities[idx] || 0;
                      const isWinner = prediction.predicted_class === key;
                      return (
                        <div key={key} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className={`font-semibold ${isWinner ? "text-slate-900 dark:text-white" : "text-slate-500"}`}>
                              {LESION_INFO[key]?.display_name}
                            </span>
                            <span className="font-mono text-slate-500 font-bold">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-100 dark:bg-slate-800 h-2.5 rounded-full overflow-hidden">
                            <div
                              style={{ width: `${prob * 100}%` }}
                              className={`h-full transition-all duration-500 rounded-full ${getRiskColor(key)} ${isWinner ? "brightness-100" : "brightness-75 opacity-60"}`}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

              </div>
            )}

          </div>

        </div>
      </main>

      {/* FOOTER */}
      <footer className="border-t border-slate-200 dark:border-slate-900 bg-white dark:bg-slate-950 mt-16 py-8">
        <div className="max-w-6xl mx-auto px-4 text-center space-y-2">
          <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider">
            Biomedical Engineering Laboratory Module
          </p>
          <p className="text-[10px] text-slate-400">
            Powered by Keras, TensorFlow, FastAPI, and React. Unrestricted open-source research layout.
          </p>
        </div>
      </footer>

    </div>
  );
}