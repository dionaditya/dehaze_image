const fs = require('fs');
const path = require('path');

// Directory to look for files
const directoryPath = './hazy';

// Read the directory
fs.readdir(directoryPath, (err, files) => {
  if (err) {
    return console.error('Unable to scan directory: ' + err);
  }

  // Filter files that contain an underscore and remove them
  files.forEach((file) => {
    if (file.includes('_')) {
      const filePath = path.join(directoryPath, file);
      fs.unlink(filePath, (err) => {
        if (err) {
          console.error(`Error deleting file ${file}: `, err);
        } else {
          console.log(`${file} was deleted`);
        }
      });
    }
  });
});
