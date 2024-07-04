from django.shortcuts import render
from .forms import UploadFileForm

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            # 모델 예측 호출 및 결과 반환
            result = predict_disease(request.FILES['file'])
            return render(request, 'predictions/result.html', {'result': result})
    else:
        form = UploadFileForm()
    return render(request, 'predictions/upload.html', {'form': form})

def handle_uploaded_file(f):
    with open('uploaded_file.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

