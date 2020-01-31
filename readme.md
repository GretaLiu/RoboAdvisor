The flask application will be solely used for APIs that handle server-side business logic. Templates are statically served to minimize server load.

# API Endpoints

1. POST `/profile`: a new user created a profile, save the payload to DB, return a success message
2. GET `/investments`: a logged in user asks for information about their existing investments
3. GET `/portfolio/<id>`: a user requests more information about a particular portfolio
4. POST `/generate-portfolio`: a user requests a new portfolio, use the payload as information to generate a portfolio
5. POST `/confirm-portfolio`: a user commits to the generated portfolio, return a success message
