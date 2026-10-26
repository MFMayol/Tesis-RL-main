import sys

from simulador_irp.interfaz_consola import iniciar

if __name__ == "__main__":
    # Evita UnicodeEncodeError si la consola no soporta los acentos; se imprime un reemplazo en su lugar.
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    raise SystemExit(iniciar())
