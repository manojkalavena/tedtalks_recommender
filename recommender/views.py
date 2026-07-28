from django.shortcuts import render
from django.http import JsonResponse
from .ml.recommendation_engine import get_engine
import json


def index(request):
    engine = get_engine()
    talks = engine.get_all_talks()
    cluster_info = engine.get_cluster_info()

    talks_list = []
    for _, row in talks.iterrows():
        talks_list.append({
            'id': int(row['id']),
            'title': str(row['title']),
            'speaker': str(row['speaker']),
            'speaker_occupation': str(row['speaker_occupation']),
            'duration': str(row['duration_fmt']),
            'views': int(row['views']),
            'tags': str(row['tags']),
            'cluster_name': str(row['cluster_name']),
            'event': str(row['event']),
            'url': str(row['url']),
        })

    clusters = cluster_info.to_dict('records')

    context = {
        'talks': json.dumps(talks_list),
        'clusters': json.dumps(clusters),
        'total_talks': len(talks_list),
    }
    return render(request, 'recommender/index.html', context)


def recommend(request, talk_id):
    engine = get_engine()
    talk = engine.get_talk_by_id(talk_id)
    if talk is None:
        return JsonResponse({'error': 'Talk not found'}, status=404)

    recs = engine.recommend(talk_id, top_n=6)

    return JsonResponse({
        'selected_talk': {
            'id': int(talk['id']),
            'title': str(talk['title']),
            'speaker': str(talk['speaker']),
            'speaker_occupation': str(talk['speaker_occupation']),
            'duration': str(talk['duration_fmt']),
            'views': int(talk['views']),
            'tags': str(talk['tags']),
            'description': str(talk['description']),
            'cluster_name': str(talk['cluster_name']),
            'event': str(talk['event']),
            'url': str(talk['url']),
        },
        'recommendations': [
            {
                'id': int(r['id']),
                'title': str(r['title']),
                'speaker': str(r['speaker']),
                'speaker_occupation': str(r.get('speaker_occupation', '')),
                'duration': str(r['duration_fmt']),
                'views': int(r['views']),
                'tags': str(r['tags']),
                'description': str(r['description']),
                'cluster_name': str(r['cluster_name']),
                'url': str(r['url']),
                'hybrid_score': round(float(r['hybrid_score']), 3),
                'content_score': round(float(r['similarity_score']), 3),
                'collab_score': round(float(r['collab_score']), 3),
            }
            for _, r in recs.iterrows()
        ]
    })


def search(request):
    query = request.GET.get('q', '').strip()
    if not query:
        return JsonResponse({'results': []})

    engine = get_engine()
    results = engine.search_talks(query, top_n=10)

    return JsonResponse({
        'results': [
            {
                'id': int(r['id']),
                'title': str(r['title']),
                'speaker': str(r['speaker']),
                'duration': str(r['duration_fmt']),
                'views': int(r['views']),
                'tags': str(r['tags']),
                'cluster_name': str(r['cluster_name']),
                'event': str(r['event']),
                'score': round(float(r['search_score']), 3),
            }
            for _, r in results.iterrows()
        ]
    })


def talk_detail(request, talk_id):
    engine = get_engine()
    talk = engine.get_talk_by_id(talk_id)
    if talk is None:
        return JsonResponse({'error': 'Not found'}, status=404)

    return JsonResponse({
        'id': int(talk['id']),
        'title': str(talk['title']),
        'speaker': str(talk['speaker']),
        'speaker_occupation': str(talk['speaker_occupation']),
        'duration': str(talk['duration_fmt']),
        'views': int(talk['views']),
        'tags': str(talk['tags']),
        'description': str(talk['description']),
        'cluster_name': str(talk['cluster_name']),
        'event': str(talk['event']),
        'url': str(talk['url']),
    })
