from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .models import ScanAnalysis
from .dl.predict import classify_image
from .cv.postprocess import process_and_segment, generate_comparison_chart

def upload_view(request):
    if request.method == 'POST' and request.FILES['scan']:
        scan = request.FILES['scan']
        fs = FileSystemStorage()
        filename = fs.save(scan.name, scan)
        file_path = fs.path(filename)
        
        # 1. Deep Learning Classification
        is_tumour, confidence = classify_image(file_path)
        
        # 2. Computer Vision Segmentation
        processed_img, area = process_and_segment(file_path)
        
        # 3. Save to DB
        analysis = ScanAnalysis.objects.create(
            image=filename,
            image_name=scan.name,
            tumour_detected=is_tumour,
            confidence_score=confidence,
            tumour_area_pixels=area
        )
        
        return render(request, 'result.html', {
            'analysis': analysis,
            'processed_img': processed_img
        })
    
    return render(request, 'upload.html')

def comparison_view(request):
    if request.method == 'POST' and 'scan1' in request.FILES and 'scan2' in request.FILES:
        fs = FileSystemStorage()
        
        # Process Image 1
        f1 = request.FILES['scan1']
        n1 = fs.save(f1.name, f1)
        path1 = fs.path(n1)
        
        # Process Image 2
        f2 = request.FILES['scan2']
        n2 = fs.save(f2.name, f2)
        path2 = fs.path(n2)
        
        # 1. DL Classification - check if tumours are actually present
        is_tumour1, conf1 = classify_image(path1)
        is_tumour2, conf2 = classify_image(path2)
        
        # 2. CV Segmentation - get tumour areas
        _, area1 = process_and_segment(path1)
        _, area2 = process_and_segment(path2)
        
        # 3. If no tumour in a scan, treat its area as 0
        if not is_tumour1:
            area1 = 0
        if not is_tumour2:
            area2 = 0
        
        # 4. Determine growth status
        no_tumour = not is_tumour1 and not is_tumour2
        if no_tumour:
            growth = 0
            growth_status = 'no_tumour'
        elif is_tumour1 and not is_tumour2:
            growth = -100
            growth_status = 'disappeared'
        elif not is_tumour1 and is_tumour2:
            growth = 100
            growth_status = 'new_tumour'
        elif area1 > 0:
            growth = ((area2 - area1) / area1) * 100
            growth_status = 'grew' if growth > 0 else ('shrank' if growth < 0 else 'no_change')
        else:
            growth = 0
            growth_status = 'no_change'
            
        chart = generate_comparison_chart(area1, area2)
        
        return render(request, 'comparison.html', {
            'area1': area1,
            'area2': area2,
            'growth': round(growth, 2),
            'chart': chart,
            'image1_name': f1.name,
            'image2_name': f2.name,
            'is_tumour1': is_tumour1,
            'is_tumour2': is_tumour2,
            'conf1': round(conf1 * 100, 1),
            'conf2': round(conf2 * 100, 1),
            'no_tumour': no_tumour,
            'growth_status': growth_status,
        })

    return render(request, 'upload.html', {'mode': 'compare'})