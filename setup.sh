#!/bin/bash

echo "Setting up Fraud Detection ML Pipeline project..."

# Initialize Git if not already initialized
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial project structure"
fi

# Add CEAP platform as submodule
if [ ! -d ceap-platform ]; then
    echo "Adding CEAP platform as submodule..."
    git submodule add https://github.com/quannh0308/customer-engagement-platform.git ceap-platform
    git commit -m "Add CEAP platform as submodule"
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create your GitHub repository"
echo "2. git remote add origin https://github.com/YOUR-USERNAME/fraud-detection-system.git"
echo "3. git push -u origin main"
echo "4. Start implementing tasks from .kiro/specs/fraud-detection-ml-pipeline/tasks.md"
