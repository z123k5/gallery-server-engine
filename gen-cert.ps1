echo "Generating certificate"
mkdir ./cert
openssl genrsa -out ./cert/private.key 2048

echo "Generating certificate request"
openssl req -new -key ./cert/private.key -out ./cert/server.csr

echo "Generating certificate"
openssl x509 -req -days 365 -in ./cert/server.csr -signkey ./cert/private.key -out ./cert/certificate.crt
echo "Done."
