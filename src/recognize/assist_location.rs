use crate::models::Mark;
use crate::models::AssistLocation;
use crate::models::ProcessedImage;
use anyhow::Result;
pub struct AssistLocationModule;

impl AssistLocationModule {
    pub fn new() -> Self {
        Self
    }

    pub fn infer(&self, processed_image: &ProcessedImage, assist_location: &AssistLocation) -> Result<AssistLocation> {
        
        
        Ok(assist_location.clone())
    }

    pub fn align_assist_location(&self, processed_image: &ProcessedImage, coordinates: &Vec<Coordinate>) -> Result<AssistLocation> {
}