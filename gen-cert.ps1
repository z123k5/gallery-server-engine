echo "Generating certificate"
mkdir ./cert
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ./cert/private.key -out ./cert/certificate.crt -config ./cert/san.cnf

echo "Done."
