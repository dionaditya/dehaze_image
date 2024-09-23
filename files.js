const fs = require('fs');
const path = require('path');

const folderA = './ohazy/gt';
const folderB = './ohazy/hazy';

const getFilesFromFolder = (folder) => {
    return fs.readdirSync(folder).map(file => path.parse(file).name);
  };
  
  // Compare the two lists and find files in A not in B
  const findMissingFiles = (filesA, filesB) => {
    return filesA.filter(file => !filesB.includes(file));
  };

const filesA = getFilesFromFolder(folderA);
const filesB = getFilesFromFolder(folderB);

const filesInANotInB = findMissingFiles(filesA, filesB);
const filesInBNotInA = findMissingFiles(filesB, filesA);

console.log('Files in A but not in B:', filesInANotInB);
console.log('Files in B but not in A:', filesInBNotInA);