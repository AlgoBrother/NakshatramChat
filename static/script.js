document.addEventListener('DOMContentLoaded', (event) => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const SpeechGrammarList = window.SpeechGrammarList || window.webkitSpeechGrammarList;

    if (SpeechRecognition && SpeechGrammarList) {
        const recognition = new SpeechRecognition();
        
        // Simplified recognition configuration
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.continuous = true;

        const resultDisplay = document.getElementById('result');
        const micIcon = document.querySelector('.mic-icon');
        const startButton = document.getElementById('start-button');
        const synth = window.speechSynthesis;

        // Chat container setup
        const chatContainer = document.createElement('div');
        chatContainer.id = 'chat-container';
        chatContainer.style.cssText = `
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 20px;
            font-family: Arial, sans-serif;
        `;
        resultDisplay.parentNode.insertBefore(chatContainer, resultDisplay.nextSibling);

        let isListening = false;
        let isProcessing = false;

        // Function to get voices
        const getVoices = () => {
            return new Promise((resolve) => {
                let voices = synth.getVoices();
                if (voices.length) {
                    resolve(voices);
                } else {
                    synth.onvoiceschanged = () => {
                        voices = synth.getVoices();
                        resolve(voices);
                    };
                }
            });
        };

        const addMessageToChat = (text, isUser = false) => {
            const messageDiv = document.createElement('div');
            messageDiv.className = isUser ? 'user-message' : 'bot-message';
            messageDiv.style.cssText = `
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
                max-width: 80%;
                ${isUser ? 'margin-left: auto; background-color: #007bff; color: white;' 
                        : 'margin-right: auto; background-color: #f1f1f1;'}
            `;
            messageDiv.textContent = text;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return messageDiv;
        };

        // Improved speech synthesis function
        const speakText = async (text) => {
            if (!text.trim()) return;

            // Get available voices
            const voices = await getVoices();
            
            // Create and configure utterance
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Find a suitable voice
            const preferredVoice = voices.find(voice => 
                voice.name.includes('Female') || 
                voice.name.includes('Google US English') ||
                voice.name.includes('Microsoft Zira')
            );

            if (preferredVoice) {
                utterance.voice = preferredVoice;
            }

            // Configure speech parameters
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;

            // Cancel any ongoing speech
            synth.cancel();

            // Return promise for speech completion
            return new Promise((resolve, reject) => {
                utterance.onend = () => {
                    isProcessing = false;
                    resolve();
                };

                utterance.onerror = (error) => {
                    console.error('Speech synthesis error:', error);
                    isProcessing = false;
                    reject(error);
                };

                // Speak the text
                try {
                    isProcessing = true;
                    synth.speak(utterance);
                    
                    // Debug log
                    console.log('Speaking:', text);
                    console.log('Using voice:', utterance.voice ? utterance.voice.name : 'Default voice');
                } catch (error) {
                    console.error('Error initiating speech:', error);
                    isProcessing = false;
                    reject(error);
                }
            });
        };

        // Process response with proper audio handling
        const processResponse = async (sentences) => {
            for (const sentence of sentences) {
                // Add message to chat
                addMessageToChat(sentence, false);
                
                try {
                    // Speak the response
                    await speakText(sentence);
                    
                    // Small delay between sentences
                    await new Promise(resolve => setTimeout(resolve, 500));
                } catch (error) {
                    console.error('Error processing response:', error);
                }
            }
            
            // Resume listening after speaking
            if (!isListening) {
                startListening();
            }
        };

        // Improved listening start function
        const startListening = () => {
            if (!isListening) {
                try {
                    recognition.start();
                    isListening = true;
                    micIcon.classList.add('mic-active');
                    resultDisplay.textContent = 'Listening...';
                    console.log('Started listening');
                } catch (error) {
                    console.error('Error starting recognition:', error);
                }
            }
        };

        // Stop listening function
        const stopListening = () => {
            if (isListening) {
                try {
                    recognition.stop();
                    isListening = false;
                    micIcon.classList.remove('mic-active');
                    console.log('Stopped listening');
                } catch (error) {
                    console.error('Error stopping recognition:', error);
                }
            }
        };

        // Handle recognition results
        recognition.onresult = (event) => {
            const transcript = event.results[event.results.length - 1][0].transcript.trim().toLowerCase();
            console.log('Recognized text:', transcript);
            
            // Add user message to chat
            addMessageToChat(transcript, true);

            // Check for wake word with more flexible matching
            if (transcript.includes('ashwini') || 
                transcript.includes('ashvini') || 
                transcript.includes('ashwini')) {
                
                console.log('Wake word detected!');
                processResponse(['How can I help you?']);
            } else if (!isProcessing) {
                // Process normal conversation
                fetch('/get_response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({input: transcript}),
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Server response:', data);
                    processResponse(data.sentences);
                })
                .catch((error) => {
                    console.error('Error:', error);
                    processResponse(['Sorry, there was an error processing your request.']);
                });
            }
        };

        // Handle recognition end
        recognition.onend = () => {
            console.log('Recognition ended');
            isListening = false;
            micIcon.classList.remove('mic-active');
            
            // Restart listening if not processing
            if (!isProcessing) {
                startListening();
            }
        };

        // Handle recognition errors
        recognition.onerror = (event) => {
            console.error('Recognition error:', event.error);
            isListening = false;
            micIcon.classList.remove('mic-active');
            
            // Restart listening after a short delay
            setTimeout(() => {
                if (!isProcessing) {
                    startListening();
                }
            }, 1000);
        };

        // Start button handler
        startButton.addEventListener('click', () => {
            if (isListening) {
                stopListening();
            } else {
                startListening();
            }
        });

        // Initial start
        startListening();
        
        // Debug audio output
        const testAudio = () => {
            console.log('Testing audio output...');
            speakText('Audio system test');
        };
        
        // Test audio on page load
        setTimeout(testAudio, 1000);
    } else {
        resultDisplay.textContent = "Sorry, your browser doesn't support speech recognition.";
    }
});