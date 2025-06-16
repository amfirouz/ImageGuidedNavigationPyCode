# ImageGuidedNavigationPyCode
To create path planning extension
1. Download PathPlanning files 
2. In 3D Slicer, go to Modules: Developer Tools > Extension Manager, and under Extension Tools, click Create Extension.
2. Choose a name and folder for the extension.
3. Under Extension Editor, click Add Module to Extension, type a name. When prompted to, click 'Yes' to load the module now. This will create a folder for the module within the extension with a .py file named <ModuleName>.py
4. Rename PathPlanning.py to <ModuleName>.py and place it in the folder, replacing the existing file.
6. Within the same folder, navigate to Resources > UI, then rename PathPlanning.ui to <ModuleName>.py and place it in the folder, replacing the existing file.
7. To use the extension you just created, go to Modules: Examples > ModuleName .
