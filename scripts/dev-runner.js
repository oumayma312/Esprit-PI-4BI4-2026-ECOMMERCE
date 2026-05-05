const { spawn } = require('child_process');
const path = require('path');

const rootDir = path.resolve(__dirname, '..');
const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';
const spawnOptions = {
  cwd: rootDir,
  stdio: 'inherit',
  shell: process.platform === 'win32'
};

const children = [
  spawn(npmCommand, ['run', 'start:ui'], spawnOptions),
  spawn(pythonCommand, ['-m', 'uvicorn', 'chatbot.api:app', '--host', '127.0.0.1', '--port', '8000', '--reload'], spawnOptions)
];

let shuttingDown = false;

const shutdown = signal => {
  if (shuttingDown) {
    return;
  }

  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) {
      child.kill(signal);
    }
  }
};

for (const child of children) {
  child.on('exit', code => {
    if (shuttingDown) {
      return;
    }

    if (typeof code === 'number' && code !== 0) {
      shutdown('SIGTERM');
      process.exit(code);
    }
  });
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
