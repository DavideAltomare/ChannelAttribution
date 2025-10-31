\name{install_pro}
\alias{install_pro}
\title{Install ChannelAttributionPro binary tailored to your OS/arch/R}

\description{
Interactively installs the \pkg{ChannelAttributionPro} package by contacting the
ChannelAttribution build service, which returns a platform-appropriate binary (or
source) package URL for your current operating system, CPU architecture, and R
version. The function prompts for either:
\itemize{
  \item a \emph{token} (recommended), or
  \item a work/university \emph{email} to request a token (the function will send the request and exit).
}
}

\usage{
install_pro()
}

\details{
On launch, the function asks you to enter your ChannelAttributionPro \emph{token}.
If you don’t have one, you can enter your work/university email; the function will
request a token for that address and stop, so you can re-run it once the token
arrives.

The function:
\enumerate{
  \item Detects your platform (\code{os}, \code{os_vers}, \code{arch}) and R minor version.
  \item Contacts the ChannelAttribution build endpoint to obtain a package URL.
  \item Installs the returned package via \code{\link[utils]{install.packages}} with \code{repos = NULL}.
  \item Optionally notifies a status endpoint with a small text payload (no JSON).
}

\strong{Optional dependencies:}
This function uses \pkg{curl} for HTTP requests. If \pkg{curl} is not available,
the function aborts with a friendly message. No other packages are required.

\strong{Environment token:}
If you prefer not to type the token, you can export an environment variable
\code{CHPRO_TOKEN} before calling the function. If the prompt is left empty and
\code{CHPRO_TOKEN} is unset, the function returns invisibly without installing.

\strong{Networking and privacy:}
The function performs outbound HTTPS requests to \url{https://app.channelattribution.io}.
It sends only the minimal parameters required to resolve the correct binary and
(optionally) a short status message. No local files are uploaded.

\strong{Interactive prompt:}
Input masking is implemented using base R (with a best-effort \code{stty -echo}
on Unix-like terminals); on some environments the input may be visible.
}

\value{
Invisibly returns \code{NULL}. Messages are printed to indicate success or failure.
}

\section{Errors and messages}{
\itemize{
  \item \emph{Invalid or expired token}: You will be asked to contact
  \email{info@channelattribution.io} or request a new token.
  \item \emph{Network/SSL issues}: The function reports a concise
  \code{network_or_ssl_error: ...}
  \item \emph{Unexpected builder response}: The function prints diagnostic
  information (platform and R version) you can forward to support.
}
}

\seealso{
\itemize{
  \item \code{\link[utils]{install.packages}}
}
}

\examples{
\dontrun{
## Typical interactive use:
install_pro()

## Non-interactive (using an environment variable):
## Sys.setenv(CHPRO_TOKEN = "<your_token_here>")
## install_pro()

## If you don't have a token at prompt:
##   - enter your work/university email to request one
##   - check your inbox (including spam)
##   - re-run install_pro() with the token
}
}

\author{ChannelAttribution Team}
\encoding{UTF-8}
\keyword{utilities}

