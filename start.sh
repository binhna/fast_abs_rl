uwsgi --socket 0.0.0.0:5555 --protocol=http -w wsgi:app --threads=1 --processes=1
