const fs = require('fs');
const path = require('path');

// Function to rename files in the 'ssrc' directory
function renameFilesInDirectory(directory) {
  fs.readdir(directory, (err, files) => {
    if (err) {
      console.error(`Error reading directory: ${err}`);
      return;
    }

    files.forEach(file => {
      const filePath = path.join(directory, file);

      // Ensure we are dealing with a file
      if (fs.statSync(filePath).isFile()) {
        const newName = file.split('_')[0] + path.extname(file);
        const newFilePath = path.join(directory, newName);

        fs.rename(filePath, newFilePath, (err) => {
          if (err) {
            console.error(`Error renaming file: ${err}`);
          } else {
            console.log(`Renamed: ${filePath} -> ${newFilePath}`);
          }
        });
      }
    });
  });
}

// Usage example
const directoryPath = './ohazy/val/hazy'; // Path to the 'ssrc' directory
renameFilesInDirectory(directoryPath);