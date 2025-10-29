# Recomendador-h-brido-de-experiencias-al-aire-libre-y-rutas
Batch layer: entrena embeddings item/user con SVD (TruncatedSVD) sobre historial de valoraciones.

Speed layer: captura valoraciones nuevas en tiempo real y crea un perfil de usuario reciente (moving window) que ajusta recomendaciones sin re-entrenar el batch.

Service layer: combina (p = batch_score, q = speed_score) con un parámetro α para dar más peso a lo reciente cuando haya nuevas valoraciones y entrega las recomendaciones finales.
