import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const clasificar = (audioPath) => {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, "../python/clasificador.py");

    const proceso = spawn("python", [pythonScript, audioPath]);

    let dataResult = "";

    proceso.stdout.on("data", (data) => {
      dataResult += data.toString();
    });

    proceso.stderr.on("data", (data) => {
      console.error("Error Python:", data.toString());
    });

    proceso.on("close", () => {
      try {
        const resultado = JSON.parse(dataResult);
        resolve(resultado); 
      } catch (err) {
        reject("Error procesando JSON: " + err);
      }
    });
  });
};
