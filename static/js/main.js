// MediAssist Frontend Application Javascript

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatContainer = document.getElementById("chatContainer");
    const typingIndicator = document.getElementById("typingIndicator");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const micBtn = document.getElementById("micBtn");
    
    const directDiagnoseForm = document.getElementById("directDiagnoseForm");
    const directModelType = document.getElementById("directModelType");
    const symptomCheckboxContainer = document.getElementById("symptomCheckboxContainer");
    const searchSymptomInput = document.getElementById("searchSymptomInput");
    const resetChecklistBtn = document.getElementById("resetChecklistBtn");
    
    const noResultsMsg = document.getElementById("noResultsMsg");
    const resultsContent = document.getElementById("resultsContent");
    const diagDisease = document.getElementById("diagDisease");
    const diagConfidenceBar = document.getElementById("diagConfidenceBar");
    const diagConfidenceText = document.getElementById("diagConfidenceText");
    const diagSeverityIcon = document.getElementById("diagSeverityIcon");
    const diagSeverityText = document.getElementById("diagSeverityText");
    const diagSpecialist = document.getElementById("diagSpecialist");
    const diagDescription = document.getElementById("diagDescription");
    const diagPrecautions = document.getElementById("diagPrecautions");
    const diagSymptomBadges = document.getElementById("diagSymptomBadges");
    const diagModelUsed = document.getElementById("diagModelUsed");
    const languageSelector = document.getElementById("languageSelector");

    let allSymptoms = [];
    let isListening = false;
    let recognition = null;

    // Translation Map for Multilingual Support
    const translations = {
        en: {
            welcome: "Hello, I'm your AI health advisor. Describe what you're experiencing (e.g. \"I have a throbbing headache, high fever, and nausea\") and we can start evaluating.",
            noSymptomWarning: "I couldn't identify any standard symptoms from your description. Could you specify what you feel? (e.g. 'I have a fever, cough, and stomach pain')",
            typing: "MediAssist is thinking...",
            emergencyAlert: "🚨 EMERGENCY WARNING: Your description suggests a potentially life-threatening condition. Please seek immediate professional medical assistance or call emergency services (911) immediately.",
            diagnosing: "Diagnosing...",
            confidence: "confidence",
            severity: "Severity Level",
            specialist: "Recommended Specialist",
            description: "Condition Description",
            precautions: "Precautionary Guidelines",
            entities: "Extracted Symptom Entities",
            clearConfirm: "Are you sure you want to clear your chat history?",
            noHistory: "No diagnostic data yet.",
            micError: "Speech recognition error occurred.",
            micNoSupport: "Speech recognition not supported in this browser."
        },
        es: {
            welcome: "Hola, soy su asesor de salud de IA. Describa lo que está experimentando (por ejemplo, \"Tengo dolor de cabeza pulsátil, fiebre alta y náuseas\") y podemos comenzar a evaluar.",
            noSymptomWarning: "No pude identificar ningún síntoma estándar en su descripción. ¿Podría especificar lo que siente? (por ejemplo, 'Tengo fiebre, tos y dolor de estómago')",
            typing: "MediAssist está pensando...",
            emergencyAlert: "🚨 ADVERTENCIA DE EMERGENCIA: Su descripción sugiere una condición potencialmente mortal. Busque asistencia médica profesional inmediata o llame a los servicios de emergencia (911) de inmediato.",
            diagnosing: "Diagnosticando...",
            confidence: "de confianza",
            severity: "Nivel de Gravedad",
            specialist: "Especialista Recomendado",
            description: "Descripción de la Condición",
            precautions: "Pautas de Precaución",
            entities: "Entidades de Síntomas Extraídas",
            clearConfirm: "¿Está seguro de que desea borrar el historial de chat?",
            noHistory: "Aún no hay datos de diagnóstico.",
            micError: "Ocurrió un error en el reconocimiento de voz.",
            micNoSupport: "El reconocimiento de voz no es compatible con este navegador."
        },
        fr: {
            welcome: "Bonjour, je suis votre conseiller de santé IA. Décrivez ce que vous ressentez (par exemple, \"J'ai un mal de tête lancinant, une forte fièvre et des nausées\") et nous pourrons commencer l'évaluation.",
            noSymptomWarning: "Je n'ai pas pu identifier de symptômes standards dans votre description. Pourriez-vous spécifier ce que vous ressentez ? (par exemple, 'J'ai de la fièvre, de la toux et des maux d'estomac')",
            typing: "MediAssist réfléchit...",
            emergencyAlert: "🚨 ALERTE D'URGENCE: Votre description suggère un état potentiellement mortel. Veuillez consulter immédiatement un médecin professionnel ou appeler les services d'urgence (911) immédiatement.",
            diagnosing: "Diagnostic en cours...",
            confidence: "de confiance",
            severity: "Niveau de Gravité",
            specialist: "Spécialiste Recommandé",
            description: "Description de l'État",
            precautions: "Mesures de Précaution",
            entities: "Entités de Symptômes Extraites",
            clearConfirm: "Êtes-vous sûr de vouloir effacer l'historique des discussions ?",
            noHistory: "Aucune donnée de diagnostic pour le moment.",
            micError: "Une erreur de reconnaissance vocale s'est produite.",
            micNoSupport: "La reconnaissance vocale n'est pas prise en charge par ce navigateur."
        }
    };

    // Current Language Context
    let currentLang = "en";

    // --- Dynamic UI Population ---

    // Fetch and render symptoms in checkbox container
    function loadSymptomChecklist() {
        fetch("/symptoms")
            .then(res => res.json())
            .then(data => {
                allSymptoms = data.symptoms;
                renderSymptoms(allSymptoms);
            })
            .catch(err => {
                console.error("Error fetching symptoms:", err);
                symptomCheckboxContainer.innerHTML = '<div class="text-danger p-3 text-center">Failed to load symptom list.</div>';
            });
    }

    function renderSymptoms(symptomsToRender) {
        symptomCheckboxContainer.innerHTML = "";
        if (symptomsToRender.length === 0) {
            symptomCheckboxContainer.innerHTML = '<div class="text-muted p-3 text-center">No symptoms match your filter.</div>';
            return;
        }
        
        symptomsToRender.forEach(sym => {
            const displayLabel = sym.replace(/_/g, ' ');
            const div = document.createElement("div");
            div.className = "symptom-item";
            
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.name = "symptoms";
            checkbox.value = sym;
            checkbox.id = `symptom_${sym}`;
            
            const label = document.createElement("label");
            label.htmlFor = `symptom_${sym}`;
            label.className = "d-flex align-items-center w-100";
            
            const span = document.createElement("span");
            span.textContent = displayLabel;
            
            label.appendChild(checkbox);
            label.appendChild(span);
            div.appendChild(label);
            symptomCheckboxContainer.appendChild(div);
        });
    }

    // Filter checklists on keyup
    searchSymptomInput.addEventListener("keyup", () => {
        const query = searchSymptomInput.value.toLowerCase().trim();
        const filtered = allSymptoms.filter(sym => 
            sym.replace(/_/g, ' ').toLowerCase().includes(query)
        );
        renderSymptoms(filtered);
    });

    // Reset checklists
    resetChecklistBtn.addEventListener("click", () => {
        const checkboxes = symptomCheckboxContainer.querySelectorAll("input[type='checkbox']");
        checkboxes.forEach(cb => cb.checked = false);
        searchSymptomInput.value = "";
        renderSymptoms(allSymptoms);
    });

    // --- Speech Recognition Module ---
    
    function initSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.log("Speech recognition not supported in this browser.");
            micBtn.style.display = "none";
            return;
        }

        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;

        recognition.onstart = () => {
            isListening = true;
            micBtn.classList.add("active");
            chatInput.placeholder = "Listening...";
        };

        recognition.onend = () => {
            isListening = false;
            micBtn.classList.remove("active");
            chatInput.placeholder = "Describe symptoms in your own words...";
        };

        recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            alert(translations[currentLang].micError + ": " + event.error);
            isListening = false;
            micBtn.classList.remove("active");
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            chatInput.value = transcript;
        };
    }

    micBtn.addEventListener("click", () => {
        if (!recognition) return;
        
        if (isListening) {
            recognition.stop();
        } else {
            // Map languages dynamically
            if (currentLang === "es") recognition.lang = "es-ES";
            else if (currentLang === "fr") recognition.lang = "fr-FR";
            else recognition.lang = "en-US";
            
            recognition.start();
        }
    });

    // --- Message Rendering ---

    function appendMessage(text, sender) {
        const wrapper = document.createElement("div");
        wrapper.className = `message-wrapper message-${sender}`;
        
        const icon = document.createElement("div");
        icon.className = "message-icon";
        icon.textContent = sender === "bot" ? "⚕️" : "👤";
        
        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        
        // Convert double asterisks to bold text in bubbles
        let formattedText = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        formattedText = formattedText.replace(/\n/g, "<br>");
        bubble.innerHTML = formattedText;
        
        wrapper.appendChild(icon);
        wrapper.appendChild(bubble);
        chatContainer.appendChild(wrapper);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Fetch message history on load
    function loadChatHistory() {
        fetch("/history")
            .then(res => res.json())
            .then(data => {
                if (data.history && data.history.length > 0) {
                    // Clear default greeting if history exists
                    chatContainer.innerHTML = "";
                    data.history.forEach(msg => {
                        appendMessage(msg.message, msg.sender);
                    });
                }
            })
            .catch(err => console.error("Error loading chat history:", err));
    }

    // --- Dashboard Updates ---

    function updateDashboard(data) {
        noResultsMsg.classList.add("d-none");
        resultsContent.classList.remove("d-none");

        // Set disease
        diagDisease.textContent = data.disease;
        
        // Set confidence
        const confPct = Math.round(data.confidence * 100);
        diagConfidenceBar.style.width = `${confPct}%`;
        diagConfidenceText.textContent = `${confPct}% ${translations[currentLang].confidence}`;

        // Set specialist
        diagSpecialist.textContent = data.specialist;

        // Set description
        diagDescription.textContent = data.description;

        // Set severity
        let severityClass = "text-warning";
        let severityIcon = "fa-triangle-exclamation";
        if (data.severity_level.toLowerCase() === "high") {
            severityClass = "text-danger text-cyan-glow";
            severityIcon = "fa-circle-xmark";
        } else if (data.severity_level.toLowerCase() === "low") {
            severityClass = "text-success";
            severityIcon = "fa-circle-check";
        }
        
        diagSeverityText.textContent = `${data.severity_level} Severity`;
        diagSeverityText.className = `fw-bold mb-0 ${severityClass}`;
        diagSeverityIcon.className = `fa-solid ${severityIcon} fs-3 ${severityClass}`;

        // Set precautions list
        diagPrecautions.innerHTML = "";
        const precautionsArr = data.precautions.split(",");
        precautionsArr.forEach(prec => {
            const li = document.createElement("li");
            li.textContent = prec.trim();
            diagPrecautions.appendChild(li);
        });

        // Set symptom tags
        diagSymptomBadges.innerHTML = "";
        data.extracted_symptoms.forEach(sym => {
            const span = document.createElement("span");
            span.className = "badge bg-glass-symptom";
            span.textContent = sym.replace(/_/g, ' ');
            diagSymptomBadges.appendChild(span);
        });

        // Set Model used
        diagModelUsed.textContent = data.model_used || "MediAssist Pipeline";
    }

    // --- Submissions and Actions ---

    // Chat form submit
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        // Render user message
        appendMessage(text, "user");
        chatInput.value = "";

        // Show typing indicator
        typingIndicator.classList.remove("d-none");
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Post chat request
        fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text })
        })
        .then(res => res.json())
        .then(data => {
            typingIndicator.classList.add("d-none");
            
            // Render bot message
            appendMessage(data.response, "bot");
            
            // If response includes diagnostic prediction, update the dashboard!
            if (data.prediction) {
                updateDashboard(data.prediction);
            }
        })
        .catch(err => {
            console.error("Chat error:", err);
            typingIndicator.classList.add("d-none");
            appendMessage("I encountered a server error. Please try again.", "bot");
        });
    });

    // Clear chat history
    clearHistoryBtn.addEventListener("click", () => {
        if (confirm(translations[currentLang].clearConfirm)) {
            fetch("/clear_history", { method: "POST" })
                .then(res => res.json())
                .then(() => {
                    chatContainer.innerHTML = "";
                    appendMessage(translations[currentLang].welcome, "bot");
                    
                    // Reset Dashboard
                    noResultsMsg.classList.remove("d-none");
                    resultsContent.classList.add("d-none");
                })
                .catch(err => console.error("Error clearing chat history:", err));
        }
    });

    // Direct Checklist Diagnose Form submit
    directDiagnoseForm.addEventListener("submit", (e) => {
        e.preventDefault();
        
        // Find checked symptoms
        const checkedBoxes = symptomCheckboxContainer.querySelectorAll("input[type='checkbox']:checked");
        if (checkedBoxes.length === 0) {
            alert(translations[currentLang].noSymptomWarning);
            return;
        }

        const selectedSymptoms = Array.from(checkedBoxes).map(cb => cb.value.replace(/_/g, ' '));
        const symptomsText = selectedSymptoms.join(", ");
        const modelType = directModelType.value;

        // Switch to diagnostic tab immediately
        const resultsTabButton = document.getElementById("results-tab");
        const triggerEl = bootstrap.Tab.getOrCreateInstance(resultsTabButton);
        triggerEl.show();

        // Show loading state on results panel
        noResultsMsg.innerHTML = `<i class="fa-solid fa-spinner fa-spin fs-1 mb-3 text-cyan"></i><h5>${translations[currentLang].diagnosing}</h5><p class="mb-0">Running medical classifier...</p>`;
        noResultsMsg.classList.remove("d-none");
        resultsContent.classList.add("d-none");

        // POST prediction request
        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: symptomsText,
                model_type: modelType,
                model_name: modelType === 'traditional' ? 'Random Forest' : 'BERT-Tiny'
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.emergency) {
                noResultsMsg.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-danger fs-1 mb-3"></i><h5 class="text-danger">Emergency Threat Detected</h5><p class="text-white">${data.warning}</p>`;
                appendMessage(data.warning, "bot");
                return;
            }
            
            // Format prediction response to match dashboard expectation
            const formattedResult = {
                disease: data.disease,
                confidence: data.confidence,
                specialist: data.specialist,
                description: data.description,
                severity_level: data.severity_level,
                precautions: data.precautions,
                extracted_symptoms: data.extracted_symptoms,
                model_used: data.model_used
            };
            
            updateDashboard(formattedResult);
        })
        .catch(err => {
            console.error("Direct prediction error:", err);
            noResultsMsg.innerHTML = '<i class="fa-solid fa-circle-xmark text-danger fs-1 mb-3"></i><h5 class="text-danger">Diagnostic Error</h5><p class="mb-0">Failed to complete request.</p>';
        });
    });

    // --- Multilingual Translations ---

    languageSelector.addEventListener("change", () => {
        currentLang = languageSelector.value;
        
        // Translate elements that are hardcoded or static
        // 1. If chat is empty except for the welcome message, translate the welcome message
        const welcomeTextEn = translations.en.welcome;
        const welcomeTextEs = translations.es.welcome;
        const welcomeTextFr = translations.fr.welcome;
        
        const messages = chatContainer.querySelectorAll(".message-wrapper");
        if (messages.length === 1 && messages[0].classList.contains("message-bot")) {
            const bubble = messages[0].querySelector(".message-bubble p");
            if (bubble) {
                bubble.innerHTML = translations[currentLang].welcome;
            }
        }
        
        // Update check-list warning placeholders and other parts
        searchSymptomInput.placeholder = currentLang === "es" ? "Filtrar síntomas..." : (currentLang === "fr" ? "Filtrer les symptômes..." : "Filter symptoms...");
        chatInput.placeholder = currentLang === "es" ? "Describa los síntomas con sus propias palabras..." : (currentLang === "fr" ? "Décrivez les symptômes avec vos propres mots..." : "Describe symptoms in your own words...");
    });

    // --- On Startup Initializations ---
    loadSymptomChecklist();
    initSpeechRecognition();
    loadChatHistory();
});
