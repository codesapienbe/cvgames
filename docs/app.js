// CVGames Web Application

// Global state
const state = {
    games: [],
    categories: [],
    activeSection: 'home',
    filters: {
        category: 'all',
        age: 'all',
        difficulty: 'all',
        search: ''
    }
};

// DOM Elements
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.nav-links a');
const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
const navLinksContainer = document.querySelector('.nav-links');
const modal = document.getElementById('game-detail-modal');
const modalBackdrop = document.querySelector('.modal-backdrop');
const modalClose = document.querySelector('.modal-close');
const featuredCarousel = document.querySelector('.featured-carousel');
const categoriesGrid = document.querySelector('.categories-grid');
const gamesGrid = document.querySelector('.games-grid');
const webstoreGrid = document.querySelector('.webstore-grid');
const categoryFilter = document.getElementById('category-filter');
const ageFilter = document.getElementById('age-filter');
const difficultyFilter = document.getElementById('difficulty-filter');
const gameSearch = document.getElementById('game-search');
const webstoreCategories = document.querySelector('.webstore-categories');
const docNavButtons = document.querySelectorAll('.doc-nav-btn');
const docPanels = document.querySelectorAll('.doc-panel');
const projectNavButtons = document.querySelectorAll('.project-nav-btn');
const projectPanels = document.querySelectorAll('.project-panel');

// Initialize the application
async function initApp() {
    await loadGamesData();
    renderFeaturedGames();
    renderCategoriesGrid();
    renderGamesGrid();
    renderWebstoreGrid();
    populateCategoryFilter();
    setupEventListeners();
    loadDocumentation();
    loadProjectInfo();
}

// Load games data from JSON
async function loadGamesData() {
    try {
        const response = await fetch('https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/46750d6641d250c00c2ec0f3c6f44d5d/344fb35f-cbbc-4103-b9e3-44b2a72e2105/a8b2f669.json');
        if (!response.ok) {
            throw new Error('Failed to load game data');
        }
        const data = await response.json();
        state.games = data.games;
        state.categories = data.metadata.categories;
    } catch (error) {
        console.error('Error loading game data:', error);
        // Fallback to use the embedded application data if fetch fails
        const fallbackData = {
            "metadata": {
                "version": "1.0",
                "last_updated": "2025-06-17",
                "total_games": 26,
                "categories": [
                    "Hand Tracking",
                    "Face Detection",
                    "Body Movement",
                    "Eye Tracking",
                    "Gesture Recognition",
                    "Utility",
                    "Classic Games",
                    "Quiz"
                ]
            },
            "games": [
                {
                    "id": 1,
                    "title": "🎯 Virtual Darts",
                    "description": "Throw virtual darts using hand gestures. Perfect for improving hand-eye coordination!",
                    "category": "Hand Tracking",
                    "age_range": "6+",
                    "difficulty": "Easy",
                    "duration": "5-15 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/vdarts",
                    "demo_video": "https://youtu.be/codesapienbe/vdarts",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Hand tracking", "Score tracking", "Multiplayer support"],
                    "emoji": "🎯",
                    "thumbnail": "virtual-darts.png"
                },
                {
                    "id": 2,
                    "title": "🪨 Rock Paper Scissors AI",
                    "description": "Play the classic game against an AI opponent using hand gestures!",
                    "category": "Hand Tracking",
                    "age_range": "3+",
                    "difficulty": "Easy",
                    "duration": "2-10 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/rps",
                    "demo_video": "https://youtu.be/codesapienbe/rps",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Hand gesture recognition", "AI opponent", "Score tracking"],
                    "emoji": "🪨",
                    "thumbnail": "rock-paper-scissors.png"
                },
                {
                    "id": 3,
                    "title": "🎨 Air Painter",
                    "description": "Draw and paint in the air using finger movements. Express your creativity!",
                    "category": "Hand Tracking",
                    "age_range": "4+",
                    "difficulty": "Easy",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/airpainter",
                    "demo_video": "https://youtu.be/codesapienbe/airpainter",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Finger tracking", "Color selection", "Save artwork"],
                    "emoji": "🎨",
                    "thumbnail": "air-painter.png"
                },
                {
                    "id": 4,
                    "title": "🏓 Virtual Pong",
                    "description": "Control the paddle with hand movements in this classic arcade game!",
                    "category": "Hand Tracking",
                    "age_range": "6+",
                    "difficulty": "Medium",
                    "duration": "5-20 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/pong",
                    "demo_video": "https://youtu.be/codesapienbe/pong",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Hand tracking", "Difficulty levels", "High scores"],
                    "emoji": "🏓",
                    "thumbnail": "virtual-pong.png"
                },
                {
                    "id": 5,
                    "title": "👀 Eye Blink Story",
                    "description": "Progress through an interactive story using eye blinks and expressions!",
                    "category": "Eye Tracking",
                    "age_range": "8+",
                    "difficulty": "Medium",
                    "duration": "15-45 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/eyeblinkstory",
                    "demo_video": "https://youtu.be/codesapienbe/eyeblinkstory",
                    "requirements": ["OpenCV", "MediaPipe", "dlib"],
                    "features": ["Eye tracking", "Interactive story", "Multiple endings"],
                    "emoji": "👀",
                    "thumbnail": "eye-blink-story.png"
                },
                {
                    "id": 6,
                    "title": "🕺 Dance Battle",
                    "description": "Follow dance moves and compete with friends using full body tracking!",
                    "category": "Body Movement",
                    "age_range": "5+",
                    "difficulty": "Hard",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/dancebattle",
                    "demo_video": "https://youtu.be/codesapienbe/dancebattle",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Full body tracking", "Multiple songs", "Score system"],
                    "emoji": "🕺",
                    "thumbnail": "dance-battle.png"
                },
                {
                    "id": 7,
                    "title": "🎪 Circus Performer",
                    "description": "Balance virtual objects and perform circus tricks using body movements!",
                    "category": "Body Movement",
                    "age_range": "6+",
                    "difficulty": "Medium",
                    "duration": "10-25 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/circusperformer",
                    "demo_video": "https://youtu.be/codesapienbe/circusperformer",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Body tracking", "Physics simulation", "Achievement system"],
                    "emoji": "🎪",
                    "thumbnail": "circus-performer.png"
                },
                {
                    "id": 8,
                    "title": "😊 Emotion Mirror",
                    "description": "Mirror different emotions and expressions in this fun face-tracking game!",
                    "category": "Face Detection",
                    "age_range": "3+",
                    "difficulty": "Easy",
                    "duration": "5-15 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/emotionmirror",
                    "demo_video": "https://youtu.be/codesapienbe/emotionmirror",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Face detection", "Emotion recognition", "Educational content"],
                    "emoji": "😊",
                    "thumbnail": "emotion-mirror.png"
                },
                {
                    "id": 9,
                    "title": "🎯 Gesture Commander",
                    "description": "Control a spaceship using hand gestures in this space adventure!",
                    "category": "Gesture Recognition",
                    "age_range": "8+",
                    "difficulty": "Hard",
                    "duration": "15-40 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/gesturecommander",
                    "demo_video": "https://youtu.be/codesapienbe/gesturecommander",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Complex gestures", "Story mode", "Leaderboards"],
                    "emoji": "🎯",
                    "thumbnail": "gesture-commander.png"
                },
                {
                    "id": 10,
                    "title": "🎵 Sound Conductor",
                    "description": "Conduct an orchestra using hand movements and create beautiful music!",
                    "category": "Gesture Recognition",
                    "age_range": "5+",
                    "difficulty": "Medium",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/soundconductor",
                    "demo_video": "https://youtu.be/codesapienbe/soundconductor",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Music creation", "Gesture control", "Record compositions"],
                    "emoji": "🎵",
                    "thumbnail": "sound-conductor.png"
                },
                {
                    "id": 11,
                    "title": "🚗 Drive Simulator",
                    "description": "Drive a virtual car using steering wheel hand gestures!",
                    "category": "Hand Tracking",
                    "age_range": "8+",
                    "difficulty": "Medium",
                    "duration": "15-45 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/drivesimulator",
                    "demo_video": "https://youtu.be/codesapienbe/drivesimulator",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Steering control", "Multiple tracks", "Speed challenges"],
                    "emoji": "🚗",
                    "thumbnail": "drive-simulator.png"
                },
                {
                    "id": 12,
                    "title": "🧩 Puzzle Master",
                    "description": "Solve puzzles using hand gestures to move pieces around!",
                    "category": "Hand Tracking",
                    "age_range": "6+",
                    "difficulty": "Medium",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/puzzlemaster",
                    "demo_video": "https://youtu.be/codesapienbe/puzzlemaster",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Drag and drop", "Multiple difficulty levels", "Timer challenges"],
                    "emoji": "🧩",
                    "thumbnail": "puzzle-master.png"
                },
                {
                    "id": 13,
                    "title": "🤸 Body Movement Challenge",
                    "description": "Use full-body movements to navigate an obstacle course and earn points!",
                    "category": "Body Movement",
                    "age_range": "8+",
                    "difficulty": "Medium",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/bodymovementchallenge",
                    "demo_video": "https://youtu.be/codesapienbe/bodymovementchallenge",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Obstacle course", "Score tracking", "Multiple levels"],
                    "emoji": "🤸"
                },
                {
                    "id": 14,
                    "title": "🧮 Virtual Calculator",
                    "description": "Solve math problems by showing number gestures to the webcam. Fun and educational!",
                    "category": "Utility",
                    "age_range": "6+",
                    "difficulty": "Easy",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/virtualcalculator",
                    "demo_video": "https://youtu.be/codesapienbe/virtualcalculator",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Math problems", "Number gestures", "Score tracking"],
                    "emoji": "🧮"
                },
                {
                    "id": 15,
                    "title": "🎭 Face Filter Fun",
                    "description": "Apply fun filters to your face in real-time for creative photo and video effects!",
                    "category": "Face Detection",
                    "age_range": "3+",
                    "difficulty": "Easy",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/facefilterfun",
                    "demo_video": "https://youtu.be/codesapienbe/facefilterfun",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Face filters", "Photo effects", "Video effects"],
                    "emoji": "🎭"
                },
                {
                    "id": 16,
                    "title": "💪 Fitness Trainer",
                    "description": "Get real-time feedback on your exercise form using pose estimation and workout guides!",
                    "category": "Body Movement",
                    "age_range": "12+",
                    "difficulty": "Medium",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/fitnesstrainer",
                    "demo_video": "https://youtu.be/codesapienbe/fitnesstrainer",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Exercise form", "Workout guides", "Real-time feedback"],
                    "emoji": "💪"
                },
                {
                    "id": 17,
                    "title": "⌨️ Virtual Keyboard",
                    "description": "Type on a virtual keyboard using finger movements in mid-air. No physical keyboard needed!",
                    "category": "Hand Tracking",
                    "age_range": "8+",
                    "difficulty": "Medium",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/virtualkeyboard",
                    "demo_video": "https://youtu.be/codesapienbe/virtualkeyboard",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Finger movements", "Virtual keyboard", "Score tracking"],
                    "emoji": "⌨️"
                },
                {
                    "id": 18,
                    "title": "🕹️ Mario Master",
                    "description": "Control Mario with hand gestures in this classic platformer experience!",
                    "category": "Classic Games",
                    "age_range": "5+",
                    "difficulty": "Hard",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/mariomaster",
                    "demo_video": "https://youtu.be/codesapienbe/mariomaster",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Hand gestures", "Classic platformer", "Score tracking"],
                    "emoji": "🕹️"
                },
                {
                    "id": 19,
                    "title": "🏓 Pong Challenge",
                    "description": "Return the ball with a virtual paddle controlled by your hand position!",
                    "category": "Classic Games",
                    "age_range": "5+",
                    "difficulty": "Easy",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/pongchallenge",
                    "demo_video": "https://youtu.be/codesapienbe/pongchallenge",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Hand tracking", "Virtual paddle", "Score tracking"],
                    "emoji": "🏓"
                },
                {
                    "id": 20,
                    "title": "❓ Gesture Quiz",
                    "description": "Answer trivia questions by selecting options with hand gestures!",
                    "category": "Quiz",
                    "age_range": "8+",
                    "difficulty": "Medium",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/gesturequiz",
                    "demo_video": "https://youtu.be/codesapienbe/gesturequiz",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Hand gestures", "Trivia questions", "Score tracking"],
                    "emoji": "❓"
                },
                {
                    "id": 21,
                    "title": "📸 Selfie Fun",
                    "description": "Take selfies with timed countdowns using simple smiles or gestures!",
                    "category": "Face Detection",
                    "age_range": "3+",
                    "difficulty": "Easy",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/selfiefun",
                    "demo_video": "https://youtu.be/codesapienbe/selfiefun",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Smile detection", "Countdown timer", "Score tracking"],
                    "emoji": "📸"
                },
                {
                    "id": 22,
                    "title": "😊 Smile Detector",
                    "description": "Play mini-games by smiling or making expressions in front of the camera!",
                    "category": "Face Detection",
                    "age_range": "3+",
                    "difficulty": "Easy",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/smiledetector",
                    "demo_video": "https://youtu.be/codesapienbe/smiledetector",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Smile detection", "Mini-games", "Score tracking"],
                    "emoji": "😊"
                },
                {
                    "id": 23,
                    "title": "🐍 Gesture Snake",
                    "description": "Control the snake with body or hand movements in this modern twist!",
                    "category": "Classic Games",
                    "age_range": "6+",
                    "difficulty": "Medium",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/gesturesnake",
                    "demo_video": "https://youtu.be/codesapienbe/gesturesnake",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Body movements", "Snake game", "Score tracking"],
                    "emoji": "🐍"
                },
                {
                    "id": 24,
                    "title": "🚀 Eye Shooter",
                    "description": "Shoot down aliens by blinking or winking in this eye-tracking shooter game!",
                    "category": "Classic Games",
                    "age_range": "8+",
                    "difficulty": "Hard",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/eyeshooter",
                    "demo_video": "https://youtu.be/codesapienbe/eyeshooter",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Eye tracking", "Alien shooting", "Score tracking"],
                    "emoji": "🚀"
                },
                {
                    "id": 25,
                    "title": "🧱 Tetris Twist",
                    "description": "Rotate and drop blocks using hand gestures in this CV-powered Tetris!",
                    "category": "Classic Games",
                    "age_range": "6+",
                    "difficulty": "Medium",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/tetristwist",
                    "demo_video": "https://youtu.be/codesapienbe/tetristwist",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Hand gestures", "Tetris game", "Score tracking"],
                    "emoji": "🧱"
                },
                {
                    "id": 26,
                    "title": "❌⭕ Eye Tic Tac Toe",
                    "description": "Play Tic Tac Toe by selecting cells with your gaze in this eye-tracking twist on the classic game!",
                    "category": "Classic Games",
                    "age_range": "5+",
                    "difficulty": "Easy",
                    "duration": "10-30 minutes",
                    "download_link": "https://github.com/codesapienbe/cvgames/eyetictactoe",
                    "demo_video": "https://youtu.be/codesapienbe/eyetictactoe",
                    "requirements": ["OpenCV", "MediaPipe"],
                    "features": ["Eye tracking", "Tic Tac Toe", "Score tracking"],
                    "emoji": "❌⭕"
                }
            ]
        };
        
        state.games = fallbackData.games;
        state.categories = fallbackData.metadata.categories;
    }
}

// Setup event listeners
function setupEventListeners() {
    // Navigation
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.getAttribute('data-section');
            navigateToSection(section);
        });
    });

    // Mobile menu toggle
    mobileMenuToggle.addEventListener('click', () => {
        navLinksContainer.classList.toggle('active');
    });

    // Modal close
    modalClose.addEventListener('click', closeModal);
    modalBackdrop.addEventListener('click', closeModal);

    // Filters
    categoryFilter.addEventListener('change', updateFilters);
    ageFilter.addEventListener('change', updateFilters);
    difficultyFilter.addEventListener('change', updateFilters);
    gameSearch.addEventListener('input', updateFilters);

    // Webstore category buttons
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('category-btn')) {
            const category = e.target.getAttribute('data-category');
            setActiveWebstoreCategory(e.target);
            filterWebstoreGames(category);
        }
    });

    // Documentation navigation
    docNavButtons.forEach(button => {
        button.addEventListener('click', () => {
            const docId = button.getAttribute('data-doc');
            setActiveDocPanel(docId);
        });
    });

    // Project navigation
    projectNavButtons.forEach(button => {
        button.addEventListener('click', () => {
            const projectId = button.getAttribute('data-project');
            setActiveProjectPanel(projectId);
        });
    });

    // Link click within pages
    document.addEventListener('click', (e) => {
        if (e.target.tagName === 'A' && e.target.hasAttribute('data-section')) {
            e.preventDefault();
            const section = e.target.getAttribute('data-section');
            navigateToSection(section);
        }
    });

    // Game card click
    document.addEventListener('click', (e) => {
        const gameCard = e.target.closest('.game-card, .featured-game-card, .store-game-card');
        if (gameCard) {
            const gameId = gameCard.getAttribute('data-id');
            if (gameId) {
                openGameDetailModal(parseInt(gameId));
            }
        }
    });
}

// Navigation
function navigateToSection(section) {
    if (section === state.activeSection) return;
    
    // Update active section
    state.activeSection = section;
    
    // Update UI
    sections.forEach(s => s.classList.remove('active'));
    document.getElementById(section).classList.add('active');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-section') === section) {
            link.classList.add('active');
        }
    });
    
    // Close mobile menu if open
    navLinksContainer.classList.remove('active');
    
    // Scroll to top
    window.scrollTo(0, 0);
}

// Render featured games
function renderFeaturedGames() {
    if (!featuredCarousel) return;
    
    // Get 4 random games
    const featuredGames = getRandomGames(4);
    
    featuredCarousel.innerHTML = featuredGames.map(game => `
        <div class="featured-game-card" data-id="${game.id}">
            <div class="featured-game-emoji">
                <span>${game.emoji}</span>
            </div>
            <div class="featured-game-info">
                <h4 class="featured-game-title">${game.title}</h4>
                <p class="featured-game-description">${game.description}</p>
                <div class="featured-game-meta">
                    <span class="featured-game-category">${game.category}</span>
                    <span class="featured-game-age">${game.age_range}</span>
                </div>
            </div>
        </div>
    `).join('');
}

// Render categories grid
function renderCategoriesGrid() {
    if (!categoriesGrid) return;
    
    // Count games per category
    const categoryCount = {};
    state.categories.forEach(category => {
        categoryCount[category] = state.games.filter(game => game.category === category).length;
    });
    
    categoriesGrid.innerHTML = state.categories.map(category => {
        // Select emoji based on category
        let emoji;
        switch(category) {
            case 'Hand Tracking': emoji = '👋'; break;
            case 'Face Detection': emoji = '😊'; break;
            case 'Body Movement': emoji = '🕺'; break;
            case 'Eye Tracking': emoji = '👀'; break;
            case 'Gesture Recognition': emoji = '✌️'; break;
            default: emoji = '🎮';
        }
        
        return `
            <div class="category-card" data-category="${category}">
                <div class="category-emoji">${emoji}</div>
                <h4 class="category-name">${category}</h4>
                <span class="category-count">${categoryCount[category]} games</span>
            </div>
        `;
    }).join('');
    
    // Add event listeners
    const categoryCards = document.querySelectorAll('.category-card');
    categoryCards.forEach(card => {
        card.addEventListener('click', () => {
            const category = card.getAttribute('data-category');
            navigateToSection('games');
            categoryFilter.value = category;
            updateFilters();
        });
    });
}

// Render games grid
function renderGamesGrid() {
    if (!gamesGrid) return;
    
    const filteredGames = filterGames();
    
    if (filteredGames.length === 0) {
        gamesGrid.innerHTML = `
            <div class="no-results">
                <p>No games found matching your filters. Try adjusting your search criteria.</p>
            </div>
        `;
        return;
    }
    
    gamesGrid.innerHTML = filteredGames.map(game => `
        <div class="game-card" data-id="${game.id}">
            <div class="game-card-header">
                <span class="game-emoji">${game.emoji}</span>
                <span class="game-difficulty difficulty-${game.difficulty.toLowerCase()}">${game.difficulty}</span>
            </div>
            <div class="game-card-body">
                <h4 class="game-title">${game.title}</h4>
                <p class="game-description">${game.description}</p>
                <div class="game-meta">
                    <span class="game-category">${game.category}</span>
                    <span class="game-age">${game.age_range}</span>
                </div>
            </div>
        </div>
    `).join('');
}

// Render webstore grid
function renderWebstoreGrid() {
    if (!webstoreGrid) return;
    
    // Initialize webstore categories
    renderWebstoreCategories();
    
    webstoreGrid.innerHTML = state.games.map(game => `
        <div class="store-game-card" data-id="${game.id}" data-category="${game.category}">
            <div class="store-game-header">
                <span class="store-game-emoji">${game.emoji}</span>
                <h4 class="store-game-title">${game.title}</h4>
            </div>
            <div class="store-game-body">
                <p class="store-game-description">${game.description}</p>
                <ul class="store-game-features">
                    ${(game.features || []).slice(0, 3).map(feature => `<li>${feature}</li>`).join('')}
                </ul>
            </div>
            <div class="store-game-footer">
                <span class="store-game-meta">${game.age_range} | ${game.difficulty}</span>
                <a href="${game.download_link}" class="btn btn--primary btn--sm" target="_blank">Download</a>
            </div>
        </div>
    `).join('');
}

// Render webstore categories
function renderWebstoreCategories() {
    if (!webstoreCategories) return;
    
    const categoryButtons = state.categories.map(category => 
        `<button class="category-btn" data-category="${category}">${category}</button>`
    ).join('');
    
    webstoreCategories.innerHTML = `
        <button class="category-btn active" data-category="all">All Games</button>
        ${categoryButtons}
    `;
}

// Filter webstore games
function filterWebstoreGames(category) {
    const storeCards = document.querySelectorAll('.store-game-card');
    
    storeCards.forEach(card => {
        if (category === 'all' || card.getAttribute('data-category') === category) {
            card.style.display = 'flex';
        } else {
            card.style.display = 'none';
        }
    });
}

// Set active webstore category
function setActiveWebstoreCategory(button) {
    const categoryButtons = document.querySelectorAll('.category-btn');
    categoryButtons.forEach(btn => btn.classList.remove('active'));
    button.classList.add('active');
}

// Filter games based on user selections
function filterGames() {
    return state.games.filter(game => {
        // Category filter
        if (state.filters.category !== 'all' && game.category !== state.filters.category) {
            return false;
        }
        
        // Age filter
        if (state.filters.age !== 'all') {
            const gameMinAge = parseInt(game.age_range);
            const filterMinAge = parseInt(state.filters.age);
            if (gameMinAge !== filterMinAge) {
                return false;
            }
        }
        
        // Difficulty filter
        if (state.filters.difficulty !== 'all' && game.difficulty !== state.filters.difficulty) {
            return false;
        }
        
        // Search filter
        if (state.filters.search) {
            const searchLower = state.filters.search.toLowerCase();
            return (
                game.title.toLowerCase().includes(searchLower) ||
                game.description.toLowerCase().includes(searchLower) ||
                game.category.toLowerCase().includes(searchLower)
            );
        }
        
        return true;
    });
}

// Update filters when user changes selections
function updateFilters() {
    state.filters.category = categoryFilter.value;
    state.filters.age = ageFilter.value;
    state.filters.difficulty = difficultyFilter.value;
    state.filters.search = gameSearch.value;
    
    renderGamesGrid();
}

// Populate category filter dropdown
function populateCategoryFilter() {
    if (!categoryFilter) return;
    
    const options = state.categories.map(category => 
        `<option value="${category}">${category}</option>`
    ).join('');
    
    categoryFilter.innerHTML = `
        <option value="all">All Categories</option>
        ${options}
    `;
}

// Open game detail modal
function openGameDetailModal(gameId) {
    const game = state.games.find(g => g.id === gameId);
    if (!game) return;
    
    const modalBody = document.querySelector('.modal-body');
    
    modalBody.innerHTML = `
        <div class="game-detail-header">
            <div class="game-detail-emoji">${game.emoji}</div>
            <div class="game-detail-title-section">
                <h3>${game.title}</h3>
                <div class="game-detail-meta">
                    <span class="game-detail-badge">${game.category}</span>
                    <span class="game-detail-badge">${game.age_range}</span>
                    <span class="game-detail-badge">${game.difficulty}</span>
                </div>
            </div>
        </div>
        
        <div class="game-detail-description">
            <p>${game.description}</p>
        </div>
        
        <div class="game-detail-info">
            <div class="game-info-item">
                <span class="game-info-label">Duration</span>
                <span class="game-info-value">${game.duration}</span>
            </div>
            <div class="game-info-item">
                <span class="game-info-label">Requirements</span>
                <span class="game-info-value">${game.requirements.join(', ')}</span>
            </div>
        </div>
        
        <div class="game-detail-features">
            <h4>Features</h4>
            <div class="features-list">
                ${game.features.map(feature => 
                    `<span class="feature-item">${feature}</span>`
                ).join('')}
            </div>
        </div>
        
        <div class="game-detail-actions">
            <a href="${game.download_link}" class="btn btn--primary" target="_blank">Download Game</a>
            <a href="${game.demo_video}" class="btn btn--secondary" target="_blank">Watch Demo</a>
        </div>
    `;
    
    modal.classList.add('active');
    modalBackdrop.classList.add('active');
    document.body.style.overflow = 'hidden';
}

// Close modal
function closeModal() {
    modal.classList.remove('active');
    modalBackdrop.classList.remove('active');
    document.body.style.overflow = '';
}

// Set active documentation panel
function setActiveDocPanel(docId) {
    docNavButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-doc') === docId) {
            btn.classList.add('active');
        }
    });
    
    docPanels.forEach(panel => {
        panel.classList.remove('active');
        if (panel.id === docId) {
            panel.classList.add('active');
        }
    });
}

// Set active project panel
function setActiveProjectPanel(projectId) {
    projectNavButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-project') === projectId) {
            btn.classList.add('active');
        }
    });
    
    projectPanels.forEach(panel => {
        panel.classList.remove('active');
        if (panel.id === projectId) {
            panel.classList.add('active');
        }
    });
}

// Load documentation content - now with immediate fallback content
async function loadDocumentation() {
    // Set fallback content immediately
    const gettingStartedContent = `
        <h4>Getting Started with CVGames 🚀</h4>
        <p>CVGames is a platform for computer vision games that can be played using just a webcam. No controllers or special equipment needed!</p>
        
        <h5>Quick Start Guide</h5>
        <ol>
            <li><strong>Browse Games:</strong> Visit the Games section to see our full catalog</li>
            <li><strong>Choose a Game:</strong> Click on any game card to see detailed information</li>
            <li><strong>Download:</strong> Click the download button to get the game from GitHub</li>
            <li><strong>Setup:</strong> Make sure your webcam is connected and working</li>
            <li><strong>Play:</strong> Launch the game and follow the on-screen instructions</li>
        </ol>
        
        <h5>System Requirements</h5>
        <ul>
            <li>Working webcam (built-in or USB)</li>
            <li>Python 3.7+ installed</li>
            <li>Adequate lighting for computer vision</li>
            <li>2-3 feet of space in front of your camera</li>
        </ul>
        
        <h5>Tips for Best Performance</h5>
        <ul>
            <li>Ensure good lighting conditions</li>
            <li>Use a plain background when possible</li>
            <li>Calibrate your camera position for optimal tracking</li>
            <li>Keep your movements smooth and deliberate</li>
        </ul>
    `;
    
    const apiDocsContent = `
        <h4>API Documentation 🔌</h4>
        <p>The CVGames API provides developers with tools to create their own computer vision games using our framework.</p>
        
        <h5>Core Components</h5>
        <ul>
            <li><strong>Vision Engine:</strong> Handles camera input and computer vision processing</li>
            <li><strong>Game Framework:</strong> Provides game loop, scoring, and UI components</li>
            <li><strong>Gesture Library:</strong> Pre-built gesture recognition systems</li>
            <li><strong>Tracking Modules:</strong> Hand, face, body, and eye tracking utilities</li>
        </ul>
        
        <h5>Basic Game Structure</h5>
        <pre><code>
import cvgames

class MyGame(cvgames.Game):
    def __init__(self):
        super().__init__()
        self.hand_tracker = cvgames.HandTracker()
    
    def update(self, frame):
        hands = self.hand_tracker.process(frame)
        # Game logic here
        
    def render(self, frame):
        # Draw game elements
        return frame
        </code></pre>
        
        <h5>Available Trackers</h5>
        <ul>
            <li><code>HandTracker</code> - Track hand landmarks and gestures</li>
            <li><code>FaceTracker</code> - Detect faces and facial expressions</li>
            <li><code>PoseTracker</code> - Full body pose estimation</li>
            <li><code>EyeTracker</code> - Eye movement and blink detection</li>
        </ul>
    `;
    
    const contributingContent = `
        <h4>Contributing to CVGames 🤝</h4>
        <p>We welcome contributions from developers of all skill levels! Here's how you can get involved:</p>
        
        <h5>Ways to Contribute</h5>
        <ul>
            <li><strong>Create New Games:</strong> Develop games using our framework</li>
            <li><strong>Improve Existing Games:</strong> Add features or fix bugs</li>
            <li><strong>Enhance Documentation:</strong> Help improve our guides and tutorials</li>
            <li><strong>Report Issues:</strong> Submit bug reports and feature requests</li>
            <li><strong>Share Ideas:</strong> Suggest new game concepts or improvements</li>
        </ul>
        
        <h5>Development Process</h5>
        <ol>
            <li>Fork the repository on GitHub</li>
            <li>Create a new branch for your feature</li>
            <li>Develop and test your changes</li>
            <li>Submit a pull request with a clear description</li>
            <li>Participate in code review</li>
            <li>Celebrate when your contribution is merged! 🎉</li>
        </ol>
        
        <h5>Coding Standards</h5>
        <ul>
            <li>Follow Python PEP 8 style guidelines</li>
            <li>Include docstrings for all functions and classes</li>
            <li>Write unit tests for new features</li>
            <li>Ensure cross-platform compatibility</li>
            <li>Use meaningful variable and function names</li>
        </ul>
        
        <h5>Game Submission Guidelines</h5>
        <ul>
            <li>Games must be playable with webcam only</li>
            <li>Include clear instructions and age ratings</li>
            <li>Provide demo videos showing gameplay</li>
            <li>Test on multiple devices and lighting conditions</li>
            <li>Include accessibility features where possible</li>
        </ul>
    `;
    
    const faqContent = `
        <h4>Frequently Asked Questions ❓</h4>
        
        <h5>General Questions</h5>
        <p><strong>Q: What equipment do I need to play CVGames?</strong><br>
        A: You only need a computer with a webcam and an internet connection. No special controllers or equipment required!</p>
        
        <p><strong>Q: Do the games work on all operating systems?</strong><br>
        A: Most games work on Windows, macOS, and Linux. Check individual game requirements for specific compatibility.</p>
        
        <p><strong>Q: Are the games free?</strong><br>
        A: Yes! All games are open-source and completely free to download and play.</p>
        
        <p><strong>Q: Can I play games without an internet connection?</strong><br>
        A: Once downloaded, most games can be played offline. However, you'll need internet to download them initially.</p>
        
        <h5>Technical Support</h5>
        <p><strong>Q: My webcam isn't working with the games. What should I do?</strong><br>
        A: First, make sure your webcam works with other applications. Check privacy settings and ensure the game has camera permissions.</p>
        
        <p><strong>Q: The tracking seems inaccurate. How can I improve it?</strong><br>
        A: Try adjusting your lighting, using a plain background, and ensuring you're 2-3 feet from the camera. Calibrate if the game offers that option.</p>
        
        <p><strong>Q: Can I use an external webcam?</strong><br>
        A: Yes! External USB webcams often provide better quality than built-in laptop cameras.</p>
        
        <h5>Development Questions</h5>
        <p><strong>Q: How do I create my own game?</strong><br>
        A: Check out our API documentation and contributing guide. We provide a framework to help you get started quickly.</p>
        
        <p><strong>Q: Can I modify existing games?</strong><br>
        A: Absolutely! All games are open-source. You can fork, modify, and redistribute them following our license terms.</p>
        
        <p><strong>Q: How do I submit my game to the platform?</strong><br>
        A: Create a pull request on our GitHub repository with your game following our contribution guidelines.</p>
    `;
    
    // Set content immediately
    document.querySelector('#getting-started .doc-content-wrapper').innerHTML = gettingStartedContent;
    document.querySelector('#api-docs .doc-content-wrapper').innerHTML = apiDocsContent;
    document.querySelector('#contributing .doc-content-wrapper').innerHTML = contributingContent;
    document.querySelector('#faq .doc-content-wrapper').innerHTML = faqContent;
    
    // Try to load from external markdown files as enhancement
    try {
        const response = await fetch('https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/46750d6641d250c00c2ec0f3c6f44d5d/ad154225-ca2b-4eaf-8583-d06cd795ff6e/65f6678f.md');
        if (response.ok) {
            const markdown = await response.text();
            const sections = parseMarkdownSections(markdown);
            
            // Only override if we get valid content
            if (sections['Getting Started']) {
                document.querySelector('#getting-started .doc-content-wrapper').innerHTML = sections['Getting Started'];
            }
            if (sections['API Documentation']) {
                document.querySelector('#api-docs .doc-content-wrapper').innerHTML = sections['API Documentation'];
            }
            if (sections['Contributing Guide']) {
                document.querySelector('#contributing .doc-content-wrapper').innerHTML = sections['Contributing Guide'];
            }
            if (sections['FAQ']) {
                document.querySelector('#faq .doc-content-wrapper').innerHTML = sections['FAQ'];
            }
        }
    } catch (error) {
        console.log('External documentation not available, using fallback content');
    }
}

// Load project information - now with immediate fallback content
async function loadProjectInfo() {
    // Set fallback content immediately
    const proposalContent = `
        <h4>CVGames Project Proposal 📝</h4>
        
        <h5>Project Overview</h5>
        <p>CVGames is an open-source initiative to create accessible computer vision games that can be played using just a webcam. Our mission is to make interactive gaming accessible to everyone, regardless of physical abilities or access to specialized equipment.</p>
        
        <h5>Project Goals</h5>
        <ul>
            <li><strong>Accessibility:</strong> Create games playable by anyone with a webcam</li>
            <li><strong>Education:</strong> Demonstrate practical applications of computer vision</li>
            <li><strong>Innovation:</strong> Explore new forms of human-computer interaction</li>
            <li><strong>Community:</strong> Build a collaborative development ecosystem</li>
            <li><strong>Open Source:</strong> Maintain transparency and encourage contributions</li>
        </ul>
        
        <h5>Target Audience</h5>
        <ul>
            <li>Families looking for interactive entertainment</li>
            <li>Educational institutions teaching computer vision</li>
            <li>Developers interested in CV applications</li>
            <li>Accessibility advocates and users</li>
            <li>Gaming enthusiasts seeking new experiences</li>
        </ul>
        
        <h5>Technical Architecture</h5>
        <p>CVGames is built on a modular architecture using:</p>
        <ul>
            <li><strong>OpenCV:</strong> Core computer vision processing</li>
            <li><strong>MediaPipe:</strong> Advanced ML-based tracking</li>
            <li><strong>Python:</strong> Primary development language</li>
            <li><strong>Web Technologies:</strong> Platform and documentation</li>
        </ul>
        
        <h5>Success Metrics</h5>
        <ul>
            <li>Number of games developed and maintained</li>
            <li>Active developer community size</li>
            <li>User engagement and feedback</li>
            <li>Educational institution adoption</li>
            <li>Accessibility impact measurements</li>
        </ul>
    `;
    
    const milestonesContent = `
        <h4>Development Milestones 🏆</h4>
        
        <h5>Phase 1: Foundation (Q1 2025) ✅</h5>
        <ul>
            <li>✅ Establish project structure and guidelines</li>
            <li>✅ Develop core framework and API</li>
            <li>✅ Create initial set of 5 games</li>
            <li>✅ Launch project website and documentation</li>
            <li>✅ Set up development tools and CI/CD</li>
        </ul>
        
        <h5>Phase 2: Expansion (Q2 2025) 🔄</h5>
        <ul>
            <li>🔄 Add 7 more games across different categories</li>
            <li>🔄 Implement advanced gesture recognition</li>
            <li>🔄 Enhance accessibility features</li>
            <li>🔄 Establish community guidelines</li>
            <li>🔄 Create tutorial videos and guides</li>
        </ul>
        
        <h5>Phase 3: Enhancement (Q3 2025) 📋</h5>
        <ul>
            <li>📋 Launch developer tools and SDK</li>
            <li>📋 Implement multiplayer capabilities</li>
            <li>📋 Add game analytics and feedback systems</li>
            <li>📋 Create educational partnership program</li>
            <li>📋 Develop mobile companion apps</li>
        </ul>
        
        <h5>Phase 4: Growth (Q4 2025) 📋</h5>
        <ul>
            <li>📋 Expand to 25+ games total</li>
            <li>📋 Launch mobile platform support</li>
            <li>📋 Implement AI-powered game recommendations</li>
            <li>📋 Create annual development conference</li>
            <li>📋 Establish sustainability funding model</li>
        </ul>
        
        <h5>Long-term Vision (2026+) 🔮</h5>
        <ul>
            <li>🔮 AR/VR integration for immersive experiences</li>
            <li>🔮 AI-generated adaptive game content</li>
            <li>🔮 Global accessibility certification</li>
            <li>🔮 Integration with learning management systems</li>
            <li>🔮 Advanced biometric feedback systems</li>
        </ul>
        
        <h5>Key Performance Indicators</h5>
        <ul>
            <li><strong>Downloads:</strong> Target 100K+ game downloads by end of 2025</li>
            <li><strong>Contributors:</strong> Engage 50+ active developers</li>
            <li><strong>Institutions:</strong> Partner with 20+ educational organizations</li>
            <li><strong>Accessibility:</strong> Support 5+ different accessibility needs</li>
        </ul>
    `;
    
    // Set content immediately
    document.querySelector('#proposal .project-content-wrapper').innerHTML = proposalContent;
    document.querySelector('#milestones .project-content-wrapper').innerHTML = milestonesContent;
    
    // Try to load from external markdown files as enhancement
    try {
        const proposalResponse = await fetch('https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/46750d6641d250c00c2ec0f3c6f44d5d/e7a4b903-cf90-4df0-b8ac-7b12d0c4aa45/8814e4a8.md');
        if (proposalResponse.ok) {
            const proposal = await proposalResponse.text();
            document.querySelector('#proposal .project-content-wrapper').innerHTML = markdownToHTML(proposal);
        }
        
        const milestonesResponse = await fetch('https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/46750d6641d250c00c2ec0f3c6f44d5d/d915113b-dcfb-462a-b7fa-a22947e20241/93bbb730.md');
        if (milestonesResponse.ok) {
            const milestones = await milestonesResponse.text();
            document.querySelector('#milestones .project-content-wrapper').innerHTML = markdownToHTML(milestones);
        }
    } catch (error) {
        console.log('External project info not available, using fallback content');
    }
}

// Parse markdown sections
function parseMarkdownSections(markdown) {
    const sections = {};
    const lines = markdown.split('\n');
    
    let currentSection = '';
    let currentContent = [];
    
    for (const line of lines) {
        if (line.startsWith('## ')) {
            // Save previous section if exists
            if (currentSection && currentContent.length > 0) {
                sections[currentSection] = markdownToHTML(currentContent.join('\n'));
            }
            
            // Start new section
            currentSection = line.substring(3).trim();
            currentContent = [];
        } else if (currentSection) {
            currentContent.push(line);
        }
    }
    
    // Save last section
    if (currentSection && currentContent.length > 0) {
        sections[currentSection] = markdownToHTML(currentContent.join('\n'));
    }
    
    return sections;
}

// Simple markdown to HTML converter
function markdownToHTML(markdown) {
    // Convert headers
    let html = markdown
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^#### (.*$)/gm, '<h4>$1</h4>');
    
    // Convert lists
    html = html.replace(/^\s*\*\s(.*$)/gm, '<li>$1</li>');
    html = html.replace(/^\s*-\s(.*$)/gm, '<li>$1</li>');
    html = html.replace(/^\s*\d+\.\s(.*$)/gm, '<li>$1</li>');
    
    // Wrap lists in ul/ol tags
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    
    // Convert inline styles
    html = html
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Convert paragraphs (any line that's not a header or list)
    html = html.replace(/^(?!<h|<ul|<\/ul|<li)[^\n](.+)/gm, '<p>$&</p>');
    
    // Convert line breaks
    html = html.replace(/\n\n/g, '<br>');
    
    return html;
}

// Get random games
function getRandomGames(count) {
    const shuffled = [...state.games].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
}

// Initialize app on page load
document.addEventListener('DOMContentLoaded', initApp);