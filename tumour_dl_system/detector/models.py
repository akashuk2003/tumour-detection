from django.db import models

class ScanAnalysis(models.Model):
    image = models.ImageField(upload_to='scans/')
    image_name = models.CharField(max_length=255)
    tumour_detected = models.BooleanField(default=False)
    confidence_score = models.FloatField(default=0.0)
    tumour_area_pixels = models.IntegerField(default=0)
    
    # For comparison functionality
    comparison_image = models.ImageField(upload_to='scans/', null=True, blank=True)
    growth_percentage = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.image_name} - {self.created_at}"