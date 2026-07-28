from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('recommend/<int:talk_id>/', views.recommend, name='recommend'),
    path('search/', views.search, name='search'),
    path('talk/<int:talk_id>/', views.talk_detail, name='talk_detail'),
]
