import { clasificar } from "../servicios/servicio_audio.js";

export const clasificarAudio = async (req, res) => {
  try {
    const audio = req.file;

    if (!audio) {
      return res.status(400).json({ error: "Debes subir un archivo de audio" });
    }

    const resultado = await clasificar(audio.path);

    return res.json({
      mensaje: "Clasificación realizada",
      ...resultado  
    });

  } catch (error) {
    console.error("Error en clasificarAudio:", error);
    res.status(500).json({ error: "Error al clasificar el audio" });
  }
};
